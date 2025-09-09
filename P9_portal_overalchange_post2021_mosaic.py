from pathlib import Path
import subprocess
post2021_changedetection = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/changedetection_processing")

post_2021_tiles = ['22LCL', '22LDL', '22NDK', '23MLQ', '18LZR', '18MUE', '18MYS', '18MZS', '18NUF', '18NVF', '18NWF', '18NWG', '18NXG', '18NXH', '18NYH', '18NZH', '19LBL', '19LCL', '19LDK', '19LDL', '19LEJ', '19LEK', '19LEL', '19LFJ', '19LFK', '19LFL', '19MCM', '20LLQ', '20LLR', '20LMQ', '20LMR', '20LNQ', '20LNR', '20LPP', '20LPQ', '20LPR', '20LQP', '20LQR', '20LRM', '20MND', '20MNE', '20MPB', '20MPS', '20MPT', '20MQA', '20MQB', '20MQC', '20MQE', '20MQS', '20MQT', '20MQU', '20MRA', '20MRB', '20MRC', '20MRS', '20MRT', '20MRU', '20NNF', '20NPF', '20NQF', '20NQG', '20NQJ', '20NRH', '21LTG', '21LTH', '21LUG', '21LUH', '21LVF', '21LWF', '21LWG', '21LXF', '21LXG', '21LXK', '21LXL', '21LYF', '21LYG', '21LYK', '21LYL', '21LZF', '21LZG', '21LZH', '21LZJ', '21LZK', '21MTM', '21MTN', '21MTP', '21MTQ', '21MTR', '21MTS', '21MTT', '21MUM', '21MUN', '21MUP', '21MUQ', '21MUR', '21MUS', '21MWN', '21MWP', '21MWQ', '21MWR', '21MXM', '21MXN', '21MXP', '21MXQ', '21MXR', '21MXU', '21MYM', '21MYN', '21MYP', '21MYQ', '21MYR', '21MYS', '21MYU', '21MZN', '21MZP', '21MZR', '21MZS', '22LBM', '22LBN', '22LBP', '22LCM', '22LCN', '22LCP', '22LDM', '22MBA', '22MBB', '22MBT', '22MBU', '22MCA', '22MCB', '22MCD', '22MCE', '22MCT', '22MCU', '22MDA', '22MDB', '22MDE', '22MDU', '22MEA', '22MEB', '22MFT', '22MGS', '22MGT', '22MGU', '22MGV', '22MHA', '22NCF', '22NCG', '22NCJ', '22NCK', '22NDF', '22NDG', '22NDH', '22NDJ', '22NEF', '22NEG', '22NEH', '23MKQ', '23MKR', '23MKS', '23MLS', '23MLT', '23MLU', '23MMS', '23MMT']
print(f"{len(post_2021_tiles)} - {post_2021_tiles}")

mosaic_input_list = []
for tile_item in post_2021_tiles:

    tile_input_raster = post2021_changedetection.joinpath(tile_item, f"{tile_item}_masked_combined_filtered_detection_sieved.tif")
    if tile_input_raster.exists():
        mosaic_input_list.append(tile_input_raster)
    print(f"creating {len(mosaic_input_list)} - {mosaic_input_list}")

output_raster = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/portal_data/post_2021/overall_treecoverchange_mosaic.tif")
# # Build gdalwarp command
if not output_raster.exists():
    cmd = [
              "gdalwarp",
              "-t_srs", "EPSG:4326",  # Target CRS
              "-r", "near",  # Resampling method
              "-ot", "Byte",
              "-tr", "0.00017966", "0.00017966",
              "-r", "max",
              "-co", "NUM_THREADS=1", # Multi-threading
              "-co", "COMPRESS=DEFLATE",
              "-co", "BIGTIFF=YES",
              "-dstnodata", "0"  # NoData value
          ] + mosaic_input_list + [str(output_raster)]

    # Run command
    subprocess.run(cmd, check=True)
