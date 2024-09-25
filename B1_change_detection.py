import datetime
from pathlib import Path
import os
from csv import reader
import glob
import numpy as np
from osgeo import gdal
from osgeo import ogr, osr
from datetime import datetime

#tiles = ['18LVQ', '18LVR', '18LWR', '18NXG', '18NXH', '18NYH', '20LLP', '20LLQ', '20LMP', '20LMQ', '20NQF', '20NQG', '20NRG', '21LYG', '21LYH', '22MBT', '22MGB']
tiles = ['20LLQ']


PATHS_LIST = "/mnt/hddarchive.nfs/amazonas_dir/aux_data/geom_data/S2_grid_AmazonBasin_mosaics_paths.csv"
S2_XYOrigins = '/mnt/hddarchive.nfs/amazonas_dir/aux_data/geom_data/S2Tiles_XYOrigin.csv'

SRC_folder_start = "/mnt/hddarchive.nfs/amazonas_dir/output/Mosaic"
output_root = Path("/mnt/hddarchive.nfs/amazonas_dir/output/Change_detection")
os.makedirs(output_root, exist_ok=True)

Stack_size = 5
VH_pattern = '*BAC*VH*.tif'
POLS = ['VH','VV']
work_dir_root = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir")


def getDate(rasterfn):
    acqdate = os.path.basename(rasterfn).split('_')[3].split('.')[0]
    return acqdate

def raster2array(rasterfn):
    raster = gdal.Open(rasterfn)
    band = raster.GetRasterBand(1).ReadAsArray().astype('float')
    return band

def array2raster(rasterfn,ORX,ORY,newRasterfn,array):
    raster = gdal.Open(str(rasterfn))
    geotransform = raster.GetGeoTransform()
    originX = ORX
    originY = ORY
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    cols = raster.RasterXSize
    rows = raster.RasterYSize

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(str(newRasterfn), cols, rows, 1, gdal.GDT_Float32,['COMPRESS=DEFLATE', 'TILED=YES','BIGTIFF=IF_NEEDED'])
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

def getDateList(files):
    Date_list = list()
    for i in range(0, len(files)):
        Date = getDate(files[i])
        Date_list.append(Date)
    return Date_list

def main():
    for tile_item in tiles:
        print(f"tile {tile_item}")

        work_dir = work_dir_root.joinpath(f"change_detection_{tile_item}")
        os.makedirs(work_dir, exist_ok=True)

        output_dir = output_root.joinpath(tile_item)
        os.makedirs(output_dir, exist_ok=True)

        ORBIT_DIRECTIONS = ['descending']
        with open(PATHS_LIST, 'r') as read_paths:
            txt_reader = reader(read_paths)
            for each_Ama_tile in txt_reader:
                Ama_tile = str(each_Ama_tile[0])
                Ama_path = str(each_Ama_tile[1])
                if Ama_tile == str(tile_item):
                    print("{} ... has paths... {}".format(tile_item, Ama_path))
                    if Ama_path == 'AD':
                        ORBIT_DIRECTIONS = ['descending', 'ascending']
        print(f"orbit direction {ORBIT_DIRECTIONS}")

        for ORBIT_DIRECTION in ORBIT_DIRECTIONS:
            output_dir_direction = output_dir.joinpath(ORBIT_DIRECTION)
            os.makedirs(output_dir_direction, exist_ok=True)

            print("Orbit Direction : {}".format(ORBIT_DIRECTION))

            if os.path.exists(Path(SRC_folder_start).joinpath(tile_item, ORBIT_DIRECTION)):
                with open(S2_XYOrigins) as WKT_Tiles:
                    WKT_lines = reader(WKT_Tiles)
                    for row in WKT_lines:
                        Current_tile = row[2]
                        if Current_tile == tile_item:
                            ORX = round(float(row[0]), 7)
                            ORY = round(float(row[1]), 7)

                MOS_folder_tile_orbit_VH = Path(SRC_folder_start).joinpath(tile_item, ORBIT_DIRECTION, VH_pattern)
                MOS_folder_tile_orbit_VV = Path(SRC_folder_start).joinpath(tile_item, ORBIT_DIRECTION,
                                                                           VH_pattern.replace('VH', 'VV'))

                Out_folder = output_dir_direction

                ####### GENERATE A LIST OF FILES TO BE ANALYZED
                VH_Files = glob.glob(str(MOS_folder_tile_orbit_VH))
                VH_Files = [VH_file for VH_file in VH_Files if not VH_file.endswith("_filled.tif")]
                VH_Files_datelist = [int(x) for x in getDateList(VH_Files)]
                VH_Files_sorted = [x for _,x in sorted(zip(VH_Files_datelist,VH_Files))]
                globals()[('VH_Files')] = VH_Files_sorted

                VV_Files = glob.glob(str(MOS_folder_tile_orbit_VV))
                VV_Files = [VV_file for VV_file in VV_Files if not VV_file.endswith("_filled.tif")]
                VV_Files_datelist = [int(x) for x in getDateList(VV_Files)]
                VV_Files_sorted = [x for _,x in sorted(zip(VV_Files_datelist,VV_Files))]
                globals()[('VV_Files')] = VV_Files_sorted

                print("vh files {}".format(VH_Files_datelist))
                print("vv files {}".format(VV_Files_datelist))

                Common_dates = sorted(list(set(VH_Files_datelist) & set(VV_Files_datelist)))

                for POL in POLS:
                    pol_r_files_list = [str(Path(SRC_folder_start).joinpath(tile_item, ORBIT_DIRECTION,
                                                                            tile_item + '_BAC_' + POL + '_' + str(
                                                                                comm_value) + '.tif')) for comm_value in
                                        Common_dates]
                    globals()[(POL + '_r_files')] = pol_r_files_list

                VH_r_files = [x for _, x in sorted(zip(Common_dates, globals()['VH_r_files']))]
                VV_r_files = [x for _, x in sorted(zip(Common_dates, globals()['VV_r_files']))]

                MASTER_GRID_TIFF = (globals()[('VV_Files')])[0]

                for POL in POLS:
                    statcubes_registry = []
                    for i in range(Stack_size, len((globals()[(POL + '_Files')])) - Stack_size + 1):
                        Date = getDate((globals()[(POL + '_Files')])[i])
                        datetime_date = datetime.strptime(Date, "%Y%m%d")
                        if not datetime_date < datetime.strptime("20180201", "%Y%m%d"): continue
                        print("CURRENTLY PROCESSING BACKSCATTER STATS for DATE: {} and for POL: {} -  ".format(Date, POL))

                        if not os.path.exists(Path(Out_folder).joinpath(POL + '_pmin_' + Date + '.tif')):
                            # ######## GENERATE PAST STACK
                            rasterarray = []
                            Stack_p_list = list()
                            for p in range(i - Stack_size, i):
                                Stack_p_list.append('raster2array(' + POL + '_Files[' + str(p) + '])')
                                rasterarray.append(raster2array(globals()[(POL + '_Files')][p]))
                            Stack_p = np.stack(rasterarray, axis=0)

                            Stack_p[Stack_p <= -99] = 'nan'
                            Stack_p_MIN = np.nanmean(Stack_p, axis=0)
                            array2raster(MASTER_GRID_TIFF, ORX, ORY,
                                         Path(Out_folder).joinpath(POL + '_pmin_' + Date + '.tif'),
                                         Stack_p_MIN)  # (globals()[(POL + '_Files')])[i]
                            print("writing file: {}".format(str(POL + '_pmin_' + Date + '.tif')))
                            statcubes_registry.append(
                                {'path': Path(Out_folder).joinpath(POL + '_pmin_' + Date + '.tif'),
                                 'orbit_direction': ORBIT_DIRECTION,
                                 'basename': str(POL + '_pmin_' + Date + '.tif')})


                            ####### GENERATE (INCLUSIVE) FUTURE STACK
                            rasterarray = []
                            Stack_f_list = list()
                            for f in range(i, i+Stack_size):
                              Stack_f_list.append('raster2array(' + POL + '_Files[' + str(f) + '])')
                              rasterarray.append(raster2array(globals()[(POL + '_Files')][f]))
                            Stack_f    = np.stack(rasterarray, axis=0)
                            
                            Stack_f[Stack_f <= -99] = 'nan'
                            Stack_f_MIN = np.nanmean(Stack_f, axis=0)
                            array2raster(MASTER_GRID_TIFF, ORX, ORY, Path(Out_folder).joinpath(POL + '_fmin_' + Date + '.tif'),Stack_f_MIN)  # (globals()[(POL + '_Files')])[i]
                            
                            print("writing file: {}".format(str(POL + '_fmin_' + Date + '.tif')))
                            statcubes_registry.append({'path':Path(Out_folder).joinpath(POL + '_fmin_' + Date + '.tif'),
                                                     'orbit_direction':ORBIT_DIRECTION,
                                                     'basename': str(POL + '_fmin_' + Date + '.tif')})

if __name__ == "__main__":
    main()