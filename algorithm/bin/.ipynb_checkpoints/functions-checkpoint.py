# -*- coding: utf-8 -*-
"""
Created on Wed May 14 00:34:41 2025

@author: naveenr
"""
import re
import datetime
from osgeo import gdal, osr
import subprocess
import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import glob

def date_from_filename(filename):
    
    # Define the pattern to search for the date in the filename
    match = re.search(r'\d{8}', filename)
    # Return the matched date
    date_str = match.group(0)
    
    year = int(date_str[:4])  # Assume year is in 21st or 22nd century
    month = int(date_str[4:6])
    day = int(date_str[6:])
    
    return datetime.date(year, month, day)


def get_raster_extent(dataset):
    
    """
    Returns the extent and resolution of a GeoTIFF file.
    
    Args:
        filepath (str): The path to the GeoTIFF file.
        
    Returns:
        tuple: A tuple containing:
            - extent (list): A list representing the extent in the format [xmin, ymin, xmax, ymax].
            - resolution (list): A list containing the x and y resolution values.
    """

    try:

        # Get geotransform and calculate extent
        geotransform = dataset.GetGeoTransform()
        x_size = dataset.RasterXSize
        y_size = dataset.RasterYSize
        xmin = geotransform[0]
        xmax = geotransform[0] + x_size * geotransform[1] + y_size * geotransform[2]
        ymin = geotransform[3] + x_size * geotransform[4] + y_size * geotransform[5]
        ymax = geotransform[3]
        extent = [xmin, ymin, xmax, ymax]
        
        # Get resolution (ensuring square pixels)
        x_res = abs(geotransform[1])
        y_res = abs(geotransform[5])
        resolution = [max(x_res, y_res), max(x_res, y_res)]
        return [extent, resolution]

    except Exception as e:
        print("Error:", e)
        return None, None
    




def raster_clip(mask_file, in_file, out_file, resampling_method='near', out_format='Float32',
                srcnodata='nan', dstnodata='nan', max_memory='2000'):
    """
    for every input in_file, get the same spatial resolution, projection, and
    extent as the input mask_file.

    output is a new raster file: out_file.
    """

    
    in0 = gdal.Open(mask_file)
    prj0 = in0.GetProjection()
    inSRS_converter = osr.SpatialReference()
    inSRS_converter.ImportFromWkt(prj0)
    prj0 = inSRS_converter.ExportToProj4()
    extent0, res0 = get_raster_extent(in0)
    extent0 = ' '.join(map(str, extent0))
    res0 = ' '.join(map(str, res0))
    size0 = '{} {}'.format(str(in0.RasterXSize), str(in0.RasterYSize))

    in1 = gdal.Open(in_file)
    prj1 = in1.GetProjection()
    inSRS_converter = osr.SpatialReference()
    inSRS_converter.ImportFromWkt(prj1)
    prj1 = inSRS_converter.ExportToProj4()
    extent1, res1 = get_raster_extent(in1)
    extent1 = ' '.join(map(str, extent1))
    res1 = ' '.join(map(str, res1))

    if (out_format=='Float32') or (out_format=='Float64'):
        predictor_num = 2
    else:
        predictor_num = 2

   
    gdal_expression = (
        'gdalwarp -s_srs "{}" -t_srs "{}" -te {} -ts {} '
        '-srcnodata {} -dstnodata {} -overwrite -multi '
        '-co COMPRESS=DEFLATE -co ZLEVEL=1 -co PREDICTOR={} -co BIGTIFF=YES '
        '-r {} -ot {} "{}" "{}"').format(
        prj1, prj0, extent0, size0, srcnodata, dstnodata, predictor_num,
        resampling_method, out_format, in_file, out_file)
    
            
            
    print(gdal_expression)
    subprocess.check_output(gdal_expression, shell=True)

    in0 = None
    in1 = None

    return



import re



def image_statistics(FILE_LIST, SITE, POL = 'HV'):
    
    
    # ASSIGN PATH FOR VRT FILES 
    VRT_PATH  = os.path.dirname(FILE_LIST[0]) + '/' + SITE + "_" + POL + "_Data.vrt"
    
    # BUILD THE VRT
    VRT_DS = gdal.BuildVRT(VRT_PATH, FILE_LIST, separate=True)
    # get the number of bands
    NUM_BANDS = VRT_DS.RasterCount
    
    DF  = pd.DataFrame(columns = ['BAND', 'BAND_NAME', 'MIN', 'MAX', 'MEAN', 'MEDIAN', 'SD', 'SKEWNESS', 'KURTOSIS'])
    for BAND in range(NUM_BANDS):
        
        DATA_INFO = VRT_DS.GetRasterBand(BAND + 1)  #ACCESS THE RASTER BAND 
        DATA = DATA_INFO.ReadAsArray()  # READ THE BAND DATA
        
        MASK = (DATA >  0.0001)  &  (~ np.isnan(DATA)) & (~ np.isinf(DATA))
        DATA_VAL = DATA[MASK == 1] # EXTRACT THE DATA BASED ON MASK
        
            
        # CALCULATE MINIMUM OF THE ENTIRE IMAGE OR ROI 
        MIN_VAL = np.nanmin(DATA_VAL).round(4)
        
        # CALCULATE MAXIMUM OF THE ENTIRE IMAGE OR ROI 
        MAX_VAL = np.nanmax(DATA_VAL).round(4)
        
        # CALCULATE MEAN OF THE ENTIRE IMAGE OR ROI 
        MEAN = np.nanmean(DATA_VAL).round(4)
        
        # CALCULATE MEDIAN OF THE ENTIRE IMAGE OR ROI 
        MEDIAN = np.round(np.nanmedian(DATA_VAL), 4)
        
        # CALCULATE SD OF THE ENTIRE IMAGE OR ROI 
        SD = np.nanstd(DATA_VAL).round(4)
        
        # CALCULATE SKEWNESS OF THE ENTIRE IMAGE OR ROI 
        SKEWNESS = np.round(skew(DATA_VAL[~np.isnan(DATA_VAL)].flatten()), 4)
        
        # CALCULATE KURTOSIS OF THE ENTIRE IMAGE OR ROI 
        KURTOSIS = np.round(kurtosis(DATA_VAL[~np.isnan(DATA_VAL)].flatten()), 4)
        
        DF2 = pd.DataFrame({'BAND': [BAND+1],'BAND_NAME':re.split(r'_HHHH|_HVHV', os.path.basename(FILE_LIST[BAND]))[0], 'MIN': [MIN_VAL], 'MAX': [MAX_VAL], 'MEAN': [MEAN], 'MEDIAN': [MEDIAN], 'SD': [SD], 'SKEWNESS':[SKEWNESS], 'KURTOSIS':[KURTOSIS]})   
        DF = pd.concat([DF, DF2], ignore_index=True)
    
    # euclidean_dist = np.zeros((num_bands, num_bands))  # To store SSIM values
    # manhattan_dist = np.zeros((num_bands, num_bands))  # To store SSIM values
    # chebyshev_dist = np.zeros((num_bands, num_bands))  # To store SSIM values
    
    # epsilon = 1e-5  # A small constant

    # # Loop through all pairs of layers
    # for i in range(num_bands):
    #     for j in range(num_bands):
            
    #         stats_image1 = df.iloc[i, :].values
    #         stats_image2 = df.iloc[j, :].values
            
    #         # Euclidean Distance
    #         euclidean_dist[i, j] = np.linalg.norm(stats_image1 - stats_image2) 
    #         # Manhattan Distance
    #         manhattan_dist[i, j] = distance.cityblock(stats_image1, stats_image2)  

    #         # Chebyshev Distance
    #         chebyshev_dist[i, j] =  distance.chebyshev(stats_image1, stats_image2)
            
            
    # euclidean_dist_df = pd.DataFrame(euclidean_dist, index=[f'Image {i+1}' for i in range(num_bands)],
    #                    columns=[f'Image {i+1}' for i in range(num_bands)])        
            
    
    # manhattan_dist_df = pd.DataFrame(manhattan_dist, index=[f'Image {i+1}' for i in range(num_bands)],
    #                    columns=[f'Image {i+1}' for i in range(num_bands)])        
            
    # chebyshev_dist_df = pd.DataFrame(chebyshev_dist, index=[f'Image {i+1}' for i in range(num_bands)],
    #                    columns=[f'Image {i+1}' for i in range(num_bands)])       
    
     
    # ssim_matrix = np.zeros((num_bands, num_bands))  # To store SSIM values

    # # Loop through all pairs of layers
    # for i in range(num_bands):
    #     for j in range(num_bands):
            

    #         # Extract the two layers
    #         layer_i = vrt_ds.GetRasterBand(i+1).ReadAsArray()
    #         layer_j = vrt_ds.GetRasterBand(j+1).ReadAsArray()
    #         if mask is not None:
    #             layer_i = layer_i[mask==1]
    #             layer_j = layer_j[mask==1]
                
    #         # Create masks to exclude NaN values
    #         mask_i = ~np.isnan(layer_i)
    #         mask_j = ~np.isnan(layer_j)

    #         # Combine masks
    #         combined_mask = mask_i & mask_j

    #         # Only compute SSIM if there are valid pixels
    #         if np.any(combined_mask):
    #             ssim_value = ssim(layer_i[combined_mask], layer_j[combined_mask],
    #                               data_range=layer_i[combined_mask].max() - layer_i[combined_mask].min())
    #         else:
    #             ssim_value = np.nan  # No valid pixels for SSIM calculation

    #         ssim_matrix[i, j] = ssim_value


    # ssim_df = pd.DataFrame(ssim_matrix, index=[f'Image {i+1}' for i in range(num_bands)],
    #                    columns=[f'Image {i+1}' for i in range(num_bands)])
    
    VRT_DS = None  # Close the dataset to ensure it is properly saved

    return DF 




   
# get the extend of the map
def GetExtent(ds):
    """ Return list of corner coordinates from a gdal Dataset """
    xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
    width, height = ds.RasterXSize, ds.RasterYSize
    xmax = xmin + width * xpixel
    ymin = ymax + height * ypixel

    return xmin, xmax, ymin, ymax



# # Function to handle flexible input from user (number, list, or range)
def filter_data(df):
    user_input = input("Enter an index (e.g., 1) / a list of indices separated by spaces or commas (e.g., 1, 5) / range of indices (e.g., 1-5): ").strip()
    
    if not user_input:
        print("\nNo filtering applied. Returning original DataFrame.")
        return df
    
    try:
        user_input_list = []
        
        # Splitting user input by comma and then processing each part
        for part in user_input.split(','):
            part = part.strip()
            
            if '-' in part:
                # Handle ranges
                start, end = map(int, part.split('-'))
                user_input_list.extend(range(start, end + 1))
            else:
                # Handle single indices
                user_input_list.append(int(part))
        
        # Filter DataFrame based on user input
        filtered_df = df[~df.index.isin(user_input_list)]
        
        if filtered_df.empty:
            print("\nFiltered DataFrame is empty.")
        else:
            print("\nFiltered DataFrame:")
           
        
        return filtered_df.reset_index(drop=True)
    
    except ValueError as e:
        print(f"\nInvalid input: {str(e)}")
        return df
    
    
    

def get_version_number(DIRECTORY, OVERWRITE = True):
    from datetime import datetime
    # LIST ALL FILES WITH VERSION NUMBER IN THE DIRECTORY
    MY_LIST = glob.glob(DIRECTORY + '\*_FILTERED_DATA.csv')
    
    # CURRENT_DATE = datetime.now().date()
    
    
    if not MY_LIST: # IF NO FILE EXISTS ASSIGN VERSION NUMBER TO 0 
        VERSION_NUMBER = 0
    else:
        DATFRAME  = pd.DataFrame()
        for FILE in MY_LIST:
            TEMP = FILE.split('_')
            DATFRAME = pd.concat([DATFRAME, pd.DataFrame({'V_N': [int(TEMP[-4])], 'Date':  [datetime.strptime(TEMP[-3], "%Y%m%d").date()] })])
        VERSION_NUMBER = DATFRAME['V_N'].max()
        
        if OVERWRITE == True:
            VERSION_NUMBER = VERSION_NUMBER + 1
        else:
            VERSION_NUMBER = VERSION_NUMBER 
        
     
     
    return VERSION_NUMBER # RETURN THE VERSION NUMBER 


def get_color_palette_nlcd():
    df = pd.DataFrame()
    df['Value'] =       [0, 11, 12,  21,  22, 23, 24, 31, 41, 42,  43, 51, 52, 71,  72, 73, 74, 81, 82, 90, 95]
    df['Description'] = ['No Data', 'Open Water', 'Perennial ice/snow', 'Developed, Open Space', 'Developed, Low Intensity', 'Developed, Medium Intensity', \
                         'Developed, High Intensity', 'Barren Land',  'Deciduous Forest', 'Evergreen Forest', 'Mixed Forest', 'Dwarf scrub', 'Shrub/scrub', \
                             'Grassland/Herbaceous', 'Sedge/Herbaceous','Lichens', 'Moss', 'Pasture/Hay', 'Cultivated Crops', 'Woody Wetlands', 'Emergent Herbaceous Wetlands']
        
    df['Palette']   = ['#000000', '#466b9f', '#d1def8', '#dec5c5', '#d99282', '#eb0000', '#ab0000', '#b3ac9f', '#68ab5f', '#1c5f2c', '#b5c58f', '#af963c', '#ccb879', '#dfdfc2', \
                   '#d1d182', '#a3cc51', '#82ba9e', '#dcd939', '#ab6c28', '#b8d9eb', '#6c9fb8']   
    return df



def write_geotiff_with_gdalcopy(ref_file, in_array, out_name):
    in0 = gdal.Open(ref_file, gdal.GA_ReadOnly)

    # if os.path.exists(out_name):
    #     os.remove(out_name)

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.CreateCopy(
        out_name,
        in0,
        0,
        ["COMPRESS=LZW", "PREDICTOR=2"],
    )
    
    if len(in_array.shape) == 2:
        
        ds.GetRasterBand(1).WriteArray(in_array)
        ds.FlushCache()  # Write to disk.
        ds = None
        in0 = None
        
    else:
        for i in range(in_array.shape[0]):
            ds.GetRasterBand(i + 1).WriteArray(in_array[i, :, :])
        ds.FlushCache()  # Write to disk.
        ds = None
        in0 = None

def create_mask_areas(input_file, out_dir, PERCENTAGE=60, IS_LATITUDE=True):
    # Open the input GeoTIFF file
    dataset = gdal.Open(input_file, gdal.GA_ReadOnly)
    if dataset is None:
        print(f"Could not open {input_file}")
        return
    
    # Get the geotransform information
    geotransform = dataset.GetGeoTransform()
    if geotransform is None:
        print("No geotransform found.")
        return
    
    

    
    
    # Determine the boundary index (x or y) based on latitude or longitude
    if IS_LATITUDE:
        boundary_index = 3  # index for latitude is 3 in the geotransform tuple
    else:
        boundary_index = 0  # index for longitude is 0 in the geotransform tuple

    # Get the raster size
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize

    # Read the data
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()

    # Calculate the dividing line in pixel coordinates
    total_pixels = np.sum(data)
    target_pixels = int(PERCENTAGE / 100 * total_pixels)

    # Find the dividing point
    accumulated_pixels = 0
    if IS_LATITUDE:
        for i in range(rows):
            accumulated_pixels += np.sum(data[i, :])
            if accumulated_pixels >= target_pixels:
                boundary_pixel = i
                break
    else:
        for j in range(cols):
            accumulated_pixels += np.sum(data[:, j])
            if accumulated_pixels >= target_pixels:
                boundary_pixel = j
                break

    # Create a mask to divide into 60% and 40%
    if IS_LATITUDE:
        mask1 = np.zeros_like(data, dtype=np.uint8)
        mask1[:boundary_pixel, :] = data[:boundary_pixel, :]  # set upper part to original mask
        mask2 = np.zeros_like(data, dtype=np.uint8)
        mask2[boundary_pixel:, :] = data[boundary_pixel:, :]  # set lower part to original mask
    else:
        mask1 = np.zeros_like(data, dtype=np.uint8)
        mask1[:, :boundary_pixel] = data[:, :boundary_pixel]  # set left part to original mask
        mask2 = np.zeros_like(data, dtype=np.uint8)
        mask2[:, boundary_pixel:] = data[:, boundary_pixel:]  # set right part to original mask

    # Apply the masks to create the divided GeoTIFFs
    masked_data1 = data * mask1
    masked_data2 = data * mask2
    
    print(len(masked_data1==1),len(masked_data2==1))
    
    # Create output filenames based on the input file
    output_file_1 = out_dir /   os.path.basename(input_file).replace('.tif', '_train.tif')
    output_file_2 = out_dir / os.path.basename(input_file).replace('.tif', '_test.tif')

    # Write out the GeoTIFFs for the divided parts
    driver = gdal.GetDriverByName('GTiff')

    # Write first part  
    half1_dataset = driver.Create(output_file_1, cols, rows, 1, band.DataType)
    half1_dataset.SetGeoTransform(geotransform)
    half1_dataset.SetProjection(dataset.GetProjection())
    half1_band = half1_dataset.GetRasterBand(1)
    half1_band.WriteArray(masked_data1)
    half1_band.FlushCache()

    # Write second part  
    half2_dataset = driver.Create(output_file_2, cols, rows, 1, band.DataType)
    half2_dataset.SetGeoTransform(geotransform)
    half2_dataset.SetProjection(dataset.GetProjection())
    half2_band = half2_dataset.GetRasterBand(1)
    half2_band.WriteArray(masked_data2)
    half2_band.FlushCache()

    # Close the datasets
    dataset = None
    half1_dataset = None
    half2_dataset = None

    print(f"Two GeoTIFF files saved: {output_file_1}, {output_file_2}")
    

def array_reshape_rolling(x_files, y_file, name='tmp', m=1, n=1, valid_min=0, chunk_size=5000):
    
    da_info = gdal.Open(y_file)
    y_data = da_info.GetRasterBand(1).ReadAsArray()
    
    # Get geotransform parameters
    geotransform = da_info.GetGeoTransform()

    # Get raster dimensions
    width = da_info.RasterXSize
    height = da_info.RasterYSize
    
    # Calculate the pixel coordinates
    x_min = geotransform[0]
    y_max = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]
    
    # Initialize lists to store latitude and longitude
    latitudes = np.zeros([height, width])
    longitudes = np.zeros([height, width])
    
    # Iterate through each pixel and calculate coordinates
    for y in range(height):
        for x in range(width):
            # Calculate pixel center coordinates
            longitudes[y, x] = x_min + (x + 0.5) * pixel_width
            latitudes[y, x] = y_max + (y + 0.5) * pixel_height
            
    # Close the dataset
    da_info = None
    
    
    ###################################################
    xdata = []
    for x_file0 in x_files:
        x_name0 = os.path.splitext(os.path.basename(x_file0))[0]
        da_info = gdal.Open(x_file0)
        rio = da_info.GetRasterBand(1).ReadAsArray()
        xdata.append(rio)
    x_data = np.stack(xdata, axis=0)
    
    
    drow, dcol = np.mgrid[-m:m + .1, -n:n + .1].astype('int64')
    y_exp = y_data[np.newaxis,: ,:]
    
    xx1 = [y_exp]
    yy1 = [y_exp]
    
    for k in range(drow.size):
        rolledx = np.roll(x_data, drow.flatten()[k], axis=1)
        rolledx = np.roll(rolledx, dcol.flatten()[k], axis=2)
        xx1.append(rolledx)
        
        rolledy = np.roll(y_exp, drow.flatten()[k], axis=1)
        rolledy = np.roll(rolledy, dcol.flatten()[k], axis=2)
        yy1.append(rolledy)

    x1_data = np.concatenate(xx1, axis=0)  
    y1_data = np.concatenate(yy1, axis=0)  
    
    x_new = np.where(y_exp > valid_min, x1_data, np.nan)
    y_new = np.where(y_exp > valid_min, y1_data, np.nan)
    
    
    multi_index = pd.MultiIndex.from_arrays([latitudes[ y_data.astype(bool)].T, longitudes[ y_data.astype(bool)].T], names=['y', 'x'])

    x1_out = x_new[:, y_data.astype(bool)].T
    x1_valid = pd.DataFrame(x1_out, index=multi_index, columns = ['target', ] + ['x_b{}_r{}'.format(i, j)
                                       for j in np.arange(drow.size) for i in np.arange(x_data.shape[0])])

    
    
    y1_out = y_new[:, y_data.astype(bool)].T
    y1_valid = pd.DataFrame(y1_out, index=multi_index,  columns =['target', ] + ['y_b{}_r{}'.format(i, j)
                                       for j in np.arange(drow.size) for i in np.arange(y_exp.shape[0])])
  
    
    x1_valid.to_csv('{}_x_valid.csv'.format(name), index=False)
    y1_valid.to_csv('{}_y_valid.csv'.format(name), index=False)
    
    return x1_valid
    


import geopandas as gpd
import rasterio
from rasterio.mask import mask
def clip_raster(shapefile_path, raster_path, output_path):
     
    gdf = gpd.read_file(shapefile_path)

    
    with rasterio.open(raster_path) as src:
        print(f"Loading raster from {raster_path}...")
    
        # --- 4. Ensure CRS Match ---
        # Check if the CRS match. If not, reproject the shapefile.
        if gdf.crs != src.crs:
            print(f"CRS mismatch. Reprojecting shapefile from {gdf.crs} to {src.crs}...")
            gdf = gdf.to_crs(src.crs)
    
        # Get the geometry of the shapefile
        # This expects a list of GeoJSON-like geometry dictionaries
        shapes = gdf.geometry.values

        # --- 5. Crop the Raster ---
        # 'crop=True' clips the raster to the extent of the shapefile
        # 'all_touched=True' includes pixels that are touched by the geometry, not just those fully within
        print("Cropping raster...")
        try:
            out_image, out_transform = mask(src, shapes, crop=True, all_touched=True)
            out_meta = src.meta.copy() # Copy the metadata from the source
        except ValueError as e:
            print(f"Error during masking: {e}")
            print("This can happen if the shapefile geometry is outside the raster extent.")
            exit(1)

        print("Raster cropped successfully.")

        # --- 6. Update Metadata and Save ---
        # Update the metadata for the new, cropped raster
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
            })

        # Write the cropped raster to a new file
        print(f"Saving cropped raster to {output_path}...")
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
            
        print("Process completed.")
        
        