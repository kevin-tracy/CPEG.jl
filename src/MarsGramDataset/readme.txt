MonteCarlo Folder: 
perturbed initial conditions and montecarlo index used by MarsGram changed for every trajectory
    model['Initial Altitude'] = random.uniform(124, 126)
    model['Initial Latitude'] = random.uniform(-1, +1)
    model['Initial Longitude'] = random.uniform(-1,+1)
    model['Year'] = 2012
    model['Month'] = 8
    model['Day'] = 6
    model['Hour'] = 5
    model['Minute'] = random.randrange(20, 30)
    model['Second'] = 0.0
    model['Number of Points'] = 1200
    model['Delta Altitude'] = -0.1 #km
    model['Delta Longitude'] = 0.00011 #deg
    model['Delta Latitude'] = 0.0000008 #deg
    model['Delta t'] = 0.0
    model['Monte Carlo'] = 1 # but try to change to more
    model['Index Montecarlo'] = i # loop index

zoffset Folder:
same as MonteCarlo Folder + randomly changed zoffset density 
    model['Z Offset'] = random.uniform(-3.25,3.25)

DustStorm Folder:
same as MonteCarlo Folder + generated global dust storm 
    model['ALS0'] = 210
    model['INTENS'] = random.uniform(0,3)
    model['RADMAX'] = 10100

max_pert Folder:
same as MonteCarlo Folder + density pertubation scale set to max (2)

all: 
same as MonteCarlo Folder + randomly changed zoffset density + generated global dust storm + density pertubation scale set to max

For Kevin: When you work on the interpolation, use these folders in order. It would be awesome if we could arrive to include all of them, but I doubt we can. The order has been created to make it increasingly more challenging.

Also, the density that you need to consider is the perturbed one (last column).
