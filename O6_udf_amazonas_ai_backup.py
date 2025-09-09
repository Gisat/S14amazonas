import functools
import sys
from typing import Dict
import xarray as xr
import numpy as np
from openeo.udf.debug import inspect

# The onnx_deps folder contains the extracted contents of the dependencies archive provided in the job options
sys.path.insert(0, "onnx_deps")
import onnxruntime as ort

DEBUG = False

MIN_VALUE = 0
MAX_VALUE = 65535

def cutoff_minmax_scale(numpy_array):
    lower_cutoff = -30
    upper_cutoff = 0
    # Step 1: Set all values less than -30 to 0
    numpy_array[numpy_array <= lower_cutoff] = lower_cutoff
    numpy_array[numpy_array > upper_cutoff] = upper_cutoff

    # Step 2: Use min-max scaling with min as -30 and max as 0
    min_val = lower_cutoff
    max_val = upper_cutoff
    scaled_arr = (((numpy_array - min_val) * (MAX_VALUE - MIN_VALUE))/(max_val - min_val))  + MIN_VALUE
    return scaled_arr



@functools.lru_cache(maxsize=5)
def load_onnx_model(model_name: str) -> ort.InferenceSession:
    """
    Loads an ONNX model from the onnx_models folder and returns an ONNX runtime session.

    Extracting the model loading code into a separate function allows us to cache the loaded model.
    This prevents the model from being loaded for every chunk of data that is processed, but only once per executor,
    which can save a lot of time, memory and ultimately processing costs.

    Should you have to download the model from a remote location, you can add the download code here, and cache the model.

    Make sure that the arguments of the method you add the @functools.lru_cache decorator to are hashable.
    Be careful with using this decorator for class methods, as the self argument is not hashable.
    In that case you can use a static method or make sure your class is hashable (more difficult): https://docs.python.org/3/faq/programming.html#faq-cache-method-calls.

    More information on this functool can be found here:
    https://docs.python.org/3/library/functools.html#functools.lru_cache
    """
    # The onnx_models folder contains the content of the model archive provided in the job options
    if DEBUG:
        return ort.InferenceSession(f"{model_name}")
    else:
        return ort.InferenceSession(f"onnx_models/{model_name}")

def preprocess_input(input_xr: xr.DataArray, ort_session: ort.InferenceSession) -> tuple:
    """
    Preprocess the input DataArray by ensuring the dimensions are in the correct order,
    reshaping it, and returning the reshaped numpy array and the original shape.
    """
    input_xr = input_xr.transpose("y", "x", "bands")
    input_shape = input_xr.shape

    inspect(data= [input_xr.shape], message="ai_numpy_array_dim")

    # Convert to numpy array
    data = input_xr.values  # Shape (y, x, bands)

    # Calculate total number of bands and time steps
    n_bands = int(input_shape[2])
    vv_vh_bandcount = int(n_bands / 2)

    inspect(data=[n_bands], message="ai_nband")
    inspect(data=[vv_vh_bandcount], message="ai_vv_vh_bandcount")

    inspect(data=[data.shape], message="ai_data_shape")

    # Extract the first 5 VH bands and 5 corresponding VV bands
    vh = data[:, :, :5]  # Shape (x, y, 5) for VH bands
    vv = data[:, :, vv_vh_bandcount:vv_vh_bandcount+5]  # Shape (x, y, 5) for VV bands

    # Subtract the corresponding VH and VV bands
    vh_vv_ratio = vh - vv  # Shape (x, y, 5)

    # Apply min-max scaling to VH, VV, and VH/VV ratio
    vh = cutoff_minmax_scale(vh)
    vv = cutoff_minmax_scale(vv)
    vh_vv_ratio = cutoff_minmax_scale(vh_vv_ratio)

    # Stack VH, VV, and VH/VV ratio along a new axis to get shape (5, x, y, 3)
    result = np.stack((vh, vv, vh_vv_ratio), axis=-1)  # Shape (x, y, 5, 3)
    result = result[:256, :256, :]

    # Transpose to get the desired shape (5, x, y, 3)
    input_np = np.transpose(result, (2, 0, 1, 3))

    input_np = np.transpose(input_np, (1, 2, 0, 3)).reshape(256, 256, 15)
    input_np = input_np[np.newaxis, ...]

    return input_np, input_shape


def run_inference(input_np: np.ndarray, ort_session: ort.InferenceSession) -> tuple:
    """
    Run inference using the ONNX runtime session and return predicted labels and probabilities.
    """
    # Get the input name expected by the ONNX model
    input_name = ort_session._inputs_meta[0].name  # Extract input name from metadata

    # Ensure input_np is a NumPy array and reshape to match model input shape
    input_np = input_np.astype(np.float32)  # Ensure correct data type

    # Run inference
    outputs = ort_session.run(None, {input_name: input_np})

    predicted_labels = outputs[0]

    return predicted_labels


def postprocess_output(predicted_labels: np.ndarray) -> tuple:
    """
    Postprocess the output by reshaping the predicted labels and probabilities into the original spatial structure.
    """

    # Reshape to match the (y, x) spatial structure
    predicted_labels = np.squeeze(predicted_labels, axis=0)  # Remove the last axis
    predicted_labels = predicted_labels*10000
    return predicted_labels


def create_output_xarray(predicted_labels: np.ndarray,
                         input_xr: xr.DataArray) -> xr.DataArray:
    """
    Create an xarray DataArray with predicted labels and probabilities stacked along the bands dimension.
    """

    return xr.DataArray(
        predicted_labels,
        dims=["bands", "y", "x"],
        coords={
            'y': input_xr.coords['y'],
            'x': input_xr.coords['x']
        }
    )


def apply_model(input_xr: xr.DataArray) -> xr.DataArray:
    """
    Run inference on the given input data using the provided ONNX runtime session.
    This method is called for each timestep in the chunk received by apply_datacube.
    """
    ## Step 1: Load the ONNX model
    ort_session = load_onnx_model("ml_model.onnx")

    ## Step 2: Preprocess the input
    # input_xr = input_xr.transpose("y", "x", "bands")
    numpy_array = input_xr.values  # Shape (y, x, bands)

    bands, dim1, dim2 = numpy_array.shape
    total_time_steps = bands // 2
    window_size = 10
    half_window = window_size // 2
    vv_vh_bandcount = total_time_steps

    DEC_array_mask_list = []
    for i in range(total_time_steps - window_size + 1):
        vh_stack_past_index = list(np.arange(i, i + half_window))
        vv_stack_past_index = list(np.arange(vv_vh_bandcount + i, vv_vh_bandcount + i + half_window))
        inspect(data=[vh_stack_past_index], message="vh_stack_past_index")
        inspect(data=[vv_stack_past_index], message="vv_stack_past_index")

        vh_stack_past = numpy_array[vh_stack_past_index]
        vv_stack_past = numpy_array[vv_stack_past_index]

        # Subtract the corresponding VH and VV bands
        vh_vv_ratio = vh_stack_past - vv_stack_past  # Shape (x, y, 5)

        # Apply min-max scaling to VH, VV, and VH/VV ratio
        vh = cutoff_minmax_scale(vh_stack_past)
        vv = cutoff_minmax_scale(vv_stack_past)
        vh_vv_ratio = cutoff_minmax_scale(vh_vv_ratio)

        # Stack VH, VV, and VH/VV ratio along a new axis to get shape (5, x, y, 3)
        result = np.stack((vh, vv, vh_vv_ratio), axis=-1)  # Shape (x, y, 5, 3)
        # result = result[:256, :256, :]

        # Transpose to get the desired shape (5, x, y, 3)
        # input_np = np.transpose(result, (2, 0, 1, 3))

        inspect(data=[result.shape], message="input np before reshape")
        input_np = np.transpose(result, (1, 2, 0, 3)).reshape(256, 256, 15)
        input_np = input_np[np.newaxis, ...]

        # input_np, input_shape = preprocess_input(input_xr, ort_session)
        ## Step 3: Perform inference
        predicted_labels = run_inference(input_np, ort_session)

        ## Step 4: Postprocess the output
        predicted_labels = postprocess_output(predicted_labels)

        DEC_array_mask_list.append(predicted_labels)


    predicted_labels_array = np.stack(DEC_array_mask_list, axis=0)
    inspect(data=[predicted_labels_array.shape], message="predicted_labels_array shape")
    ## Step 5: Create the output xarray
    return create_output_xarray(predicted_labels_array, input_xr)


def apply_datacube(cube: xr.DataArray, context: Dict) -> xr.DataArray:
    """
    Function that is called for each chunk of data that is processed.
    The function name and arguments are defined by the UDF API.

    More information can be found here:
    https://open-eo.github.io/openeo-python-client/udf.html#udf-function-names-and-signatures

    CAVEAT: Some users tend to extract the underlying numpy array and preprocess it for the model using Numpy functions.
        The order of the dimensions in the numpy array might not be the same for each back-end or when running a udf locally,
        which can lead to unexpected results.

        It is recommended to use the named dimensions of the xarray DataArray to avoid this issue.
        The order of the dimensions can be changed using the transpose method.
        While it is a better practice to do preprocessing using openeo processes, most operations are also available in Xarray.
    """
    # Define how you want to handle nan values
    cube = cube.fillna(0)

    # inspect(data=[cube.shape], message="ai_data_shape")
    # Prepare the input
    cube = cube.astype(np.float32)

    # Apply the model for each timestep in the chunk
    output_data = apply_model(cube)

    return output_data