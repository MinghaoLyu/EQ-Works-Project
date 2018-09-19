# EQ-Works-Project
EQ Works Work Sample

Cleanup: 
A sample dataset of request logs is given in data/DataSample.csv. We consider records that have identical geoinfo and timest as suspicious. Please clean up the sample dataset by filtering out those suspicious request records.

Label: 
assign each request (from data/DataSample.csv) to one of the POI locations (from data/POIList.csv) that has minimum distance to the request location.

Analysis:
With respect to each POI location, calculate the average and standard deviation of distance between the POI location to each of its assigned requests.
Draw circles that are centered at each of the POIs, each includes all requests assigned to the POI location. Find out the radius and density (i.e. request count/circle area) for each POI location.

Model:
To visualize the popularity of the POI locations, we need to map the density of POIs into a scale from -10 to 10. Please give a mathematical model to implement the mapping. (Hint: please take consideration about extreme case/outliers, and aim to be more sensitive around mean average to give maximum visual differentiability.)
Try to come up with some reasonable hypotheses regarding POIs, state all assumptions, testing steps and conclusions. Include this as a text file (with a name bonus) in your final submission
