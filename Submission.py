# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 21:30:59 2018

@author: User
"""

#import packages
from pyspark.sql import SparkSession

from pyspark.sql.functions import *

from numpy import pi

import numpy as np

#define a sparksession
spark = SparkSession.builder.appName('spark').getOrCreate()

#import data
geo = spark.read.csv('C:/Users/User/Desktop/DataSample.csv',header=True,inferSchema=True)

poi = spark.read.csv('C:/Users/User/Desktop/POIList.csv',header = True,inferSchema=True)

geo.printSchema()

poi.printSchema()

#0 cleanup, filter out suspicious records
geo = geo.dropDuplicates(subset = [' TimeSt','Latitude','Longitude'])

geo.count()

#1 Label define a function to calculate the distance between coordinates
def distance(lon1, lat1, lon2, lat2): 
    lon1=toRadians(lon1)
    lat1=toRadians(lat1)
    lon2=toRadians(lon2)
    lat2=toRadians(lat2)
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r * 1000

#1 Label calculate minimum distance
poi = poi.select('POIID',col(" Latitude").alias("poiLatitude"), col("Longitude").alias("poiLongitude"))

geo = geo.crossJoin(poi)

geo =geo.withColumn("distance(m)", distance(geo['Longitude'],geo['Latitude'], geo['poiLongitude'],geo['poiLatitude']))

geo_distance = geo.groupBy(col('_ID').alias('ID')).min('distance(m)')

geo = geo.join(geo_distance, (geo['_ID'] == geo_distance['ID']) & (geo['distance(m)'] == geo_distance['min(distance(m))'])).drop('ID').drop('min(distance(m))')


#2 Analysis average and standard deviation
avg = geo.groupBy('POIID').agg(avg('distance(m)'))

avg = avg.orderBy('avg(distance(m))')

avg.show()

std = geo.groupBy('POIID').agg(stddev('distance(m)'))

std = std.orderBy('stddev_samp(distance(m))')

std.show()

#2 Analysis radius and density
radius = geo.groupBy('POIID').agg(max('distance(m)').alias('Radius'),count('distance(m)').alias('Count'))

radius_density = radius.withColumn('Density', radius['Count']/radius['Radius']**2*pi)

radius_density.show()

#3 Model 
outliers1 = geo.approxQuantile('distance(m)',[0.25],0.05)
outliers3 = geo.approxQuantile('distance(m)',[0.75],0.05)
IQR = outliers3[0] - outliers3[0]

upper = outliers3[0] + 1.5*IQR
lower = outliers1[0] - 1.5-IQR

geo_outliers = geo[ (geo['distance(m)'] > lower) & (geo['distance(m)'] < upper)]
geo_outliers = geo_outliers.groupBy('POIID').agg(max('distance(m)').alias('Radius'), count('distance(m)').alias('Count'))
geo_outliers = geo_outliers.withColumn('Density', geo_outliers['Count']/geo_outliers['Radius']**2**pi)

geo_outliers.createOrReplaceTempView('Model')

max_density = spark.sql('SELECT MAX(Density) FROM Model').toPandas()['max(Density)'][0]
min_density = spark.sql('SELECT MIN(Density) FROM Model').toPandas()
max_density = max_density['max(Density)'][0]
min_density = min_density['min(Density)'][0]

max_density.show()
min_density.show()

#Implement model
geo_outliers.withColumn('Model', 20*(geo_outliers['Density'] - min_density)/(max_density-min_density)-10).show()


