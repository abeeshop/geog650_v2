#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import geopandas as gpd
from pysheds.grid import Grid
import mplleaflet
import rasterio as rio
import seaborn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from rasterio.plot import show
import cartopy.crs as ccrs
import fiona
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#open DEM
with rio.open(r"E:\ACO\Masters_AB\wshed_delineation\govBC_DEM.tif") as dem_src:
    img_extent = (rio.plot.plotting_extent(dem_src))
    grid = dem_src.read(1)
display(grid)


# In[3]:


grid = Grid.from_raster(r"E:\ACO\Masters_AB\wshed_delineation\govBC_DEM.tif")
dem = grid.read_raster(r"E:\ACO\Masters_AB\wshed_delineation\govBC_DEM.tif")
dem.nodata


# In[4]:


#open the raster dataset
tsi_raster = rio.open(r"E:\ACO\Masters_AB\wshed_delineation\govBC_DEM.tif")
bounds = tsi_raster.bounds
print(bounds)
#rio.plot.show(tsi_raster)


# In[5]:


#read raster to an array
tsi_array = tsi_raster.read(1)
print(tsi_array.shape)
tsi_ma = np.ma.masked_less(tsi_array,-32767)


# In[ ]:





# In[6]:


def plot_raster_array(raster, masked_array):
    #set axes and figure
    img_extent = (rio.plot.plotting_extent(raster))
    print(img_extent)
    plt.figure(figsize=(10,5))
    #proj = ccrs.UTM(zone="9")
    #globe = ccrs.Globe(ellipse="GRS80")
    #crs = ccrs.Geodetic(globe=globe)
    
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    #ax.set_extent(img_extent, crs = ccrs.UTM(zone="9"))
    ax.set_extent(img_extent, crs = ccrs.PlateCarree())
    

    #use plt.imshow, and specify the geospatial extent of the data (in the same CRS as your axes!)
    plt.imshow(masked_array, 
               extent = img_extent,
               cmap='magma')

    # create a colorbar
    cb = plt.colorbar()
    cb.set_label('Elevation (m)')

    #To add utm zone labels
    def label_utm_grid():
        ''' Warning: should only use with small area UTM maps '''
        ax = plt.gca()    
        for val,label in zip(ax.get_xticks(), ax.get_xticklabels()):
            label.set_text(str(val))
            label.set_position((val,0))  

        for val,label in zip(ax.get_yticks(), ax.get_yticklabels()):   
            label.set_text(str(val))
            label.set_position((0,val))  

        plt.tick_params(bottom=True,top=True,left=True,right=True,
                labelbottom=True,labeltop=False,labelleft=True,labelright=False)

        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        plt.grid(True)
    label_utm_grid()

dem_plot = plot_raster_array(tsi_raster,tsi_ma)


# In[7]:


grid = Grid.from_raster((r"E:\ACO\Masters_AB\wshed_delineation\govBC_DEM.tif"))
dem = grid.read_raster((r"E:\ACO\Masters_AB\wshed_delineation\govBC_DEM.tif"))

#Fill pits in DEM
pit_filled_dem = grid.fill_pits(dem)

# Fill depressions in DEM
flooded_dem = grid.fill_depressions(pit_filled_dem)
    
# Resolve flats in DEM
inflated_dem = grid.resolve_flats(flooded_dem)
#inflated_dem = flooded_dem


# In[8]:


# Determine D8 flow directions from DEM
# ----------------------
# Specify directional mapping
dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    
# Compute flow directions
# -------------------------------------
fdir = grid.flowdir(inflated_dem, dirmap=dirmap)


# In[9]:


fig = plt.figure(figsize=(8,6))
fig.patch.set_alpha(0)

plt.imshow(fdir, extent=grid.extent, cmap='viridis', zorder=2)
boundaries = ([0] + sorted(list(dirmap)))
plt.colorbar(boundaries= boundaries,
             values=sorted(dirmap))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Flow direction grid', size=14)
plt.grid(zorder=-1)
plt.tight_layout()


# In[10]:


# Calculate flow accumulation
# --------------------------
acc = grid.accumulation(fdir, dirmap=dirmap)

fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)
plt.grid('on', zorder=0)
im = ax.imshow(acc, extent=grid.extent, zorder=2,
               cmap='cubehelix',
               norm=colors.LogNorm(1, acc.max()),
               interpolation='bilinear')
plt.colorbar(im, ax=ax, label='Upstream Cells')
plt.title('Flow Accumulation', size=14)
plt.xlabel('Easting')
plt.ylabel('Northing')
plt.tight_layout()


# In[18]:


# Delineate a catchment
# ---------------------
# Specify pour point
#x, y = 683217.6189, 5577729.9722
x, y = -126.42544556, 50.32191086
#x, y = -126.42549617, 50.32266639
#x, y = -126.42583287, 50.32314352

# Snap pour point to high accumulation cell
x_snap, y_snap = grid.snap_to_mask(acc > 100, (x, y))

# Delineate the catchment
catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, 
                       xytype='coordinate')

# Crop and plot the catchment
# ---------------------------
# Clip the bounding box to the catchment
grid.clip_to(catch)
clipped_catch = grid.view(catch)

# Plot the catchment
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)

plt.grid('on', zorder=0)
im = ax.imshow(np.where(clipped_catch, clipped_catch, np.nan), extent=grid.extent,
               zorder=1, cmap='Greys_r')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Delineated Catchment', size=14)


# In[19]:


# Calling grid.polygonize without arguments will default to the catchment mask
shapes = grid.polygonize()

schema = {
    'geometry': 'Polygon',
    'properties': {'LABEL': 'float:16'}
}

with fiona.open('catchment_1.shp', 'w',
                driver='ESRI Shapefile',
                crs=grid.crs.srs,
                schema=schema) as c:
    i = 0
    for shape, value in shapes:
        rec = {}
        rec['geometry'] = shape
        rec['properties'] = {'LABEL' : str(value)}
        rec['id'] = str(i)
        c.write(rec)
        i += 1

shp = gpd.read_file('catchment_1.shp')
shp
fig, ax = plt.subplots(figsize=(6,6))
shp.plot(ax=ax)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Catchment polygon')


# In[20]:


branches = grid.extract_river_network(fdir, acc > 200, dirmap=dirmap)

schema = {
    'geometry': 'LineString',
    'properties': {}
}

with fiona.open('rivers_1.shp', 'w',
                driver='ESRI Shapefile',
                crs=grid.crs.srs,
                schema=schema) as c:
    i = 0
    for branch in branches['features']:
        rec = {}
        rec['geometry'] = branch['geometry']
        rec['properties'] = {}
        rec['id'] = str(i)
        c.write(rec)
        i += 1

sns.set_palette('husl')
fig, ax = plt.subplots(figsize=(8.5,6.5))

plt.xlim(grid.bbox[0], grid.bbox[2])
plt.ylim(grid.bbox[1], grid.bbox[3])
ax.set_aspect('equal')

for branch in branches['features']:
    line = np.asarray(branch['geometry']['coordinates'])
    plt.plot(line[:, 0], line[:, 1])
    
_ = plt.title('D8 channels', size=14)

