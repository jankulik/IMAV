from osgeo import osr, gdal

ds = gdal.Open("data/orthophoto.tif")
old_cs = osr.SpatialReference()
old_cs.ImportFromWkt(ds.GetProjectionRef())

wgs84_wkt = """
GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.01745329251994328,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4326"]]"""
new_cs = osr.SpatialReference()
new_cs.ImportFromWkt(wgs84_wkt)

transform = osr.CoordinateTransformation(old_cs, new_cs)

width = ds.RasterXSize
height = ds.RasterYSize
gt = ds.GetGeoTransform()
minx = gt[0]
miny = gt[3] + width * gt[4] + height * gt[5]

pixel_x = 1892
pixel_y = 1783

print("x:", gt[0] + pixel_x * gt[1])
print("y:", gt[3] + pixel_y * gt[5])

lat, lon, _ = transform.TransformPoint(minx, miny)
print(lat)
print(lon)
