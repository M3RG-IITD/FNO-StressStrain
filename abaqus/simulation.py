
# Crack simulation using Stress Intensity Factor
# https://iopscience.iop.org/article/10.1088/2399-1984/ab36f0/pdf (ARTICLE FOR MATERIAL PROPERTY)
from abaqusConstants import *
from caeModules import *
import abaqus
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
import random


def getPyScriptPath():
    pyName = abaqus.getCurrentScriptPath()
    print pyName
    return os.path.dirname(pyName)

def getPyScriptName():
    pyName = abaqus.getCurrentScriptPath()
    pyName = pyName.replace('/','\\')
    toReturn = pyName[pyName.rfind('\\')+1:pyName.rfind('.py')]
    return toReturn

def equals(a,b):
    tol = 5e-5*(abs(a)+abs(b))/2
    if(abs(a-b)<=tol):
        return 1
    else:
        return 0
def isEdgeExternal(instance,edge,sizeRVE):
    currentCentroid=edge.pointOn
    isStraight = 0
    try:
        currentEdge.getCurvature(0)
    except:
        isStraight = 1
    if(isStraight==0):
        return 0
    if(equals(currentCentroid[0][0],sizeRVE[0]) or equals(currentCentroid[0][1],sizeRVE[1]) or equals(currentCentroid[0][0],0) or equals(currentCentroid[0][1],0)):
        vert = edge.getVertices()
        if(not len(vert)==2):
            return 0
        point1 = instance.vertices[vert[0]].pointOn[0]
        point2 = instance.vertices[vert[1]].pointOn[0]
        if(equals(point1[0],0) and equals(point2[0],0)):
            return 1
        elif(equals(point1[1],0) and equals(point2[1],0)):
            return 1
        elif(equals(point1[0],sizeRVE[0]) and equals(point2[0],sizeRVE[0])):
            return 1
        elif(equals(point1[1],sizeRVE[1]) and equals(point2[1],sizeRVE[1])):
            return 1
        else:
            return 0
    else:
        return 0

def main():
    # Units: N, mm, s
    pyPath = getPyScriptPath()
    if(pyPath==''):
        raise AbaqusException, 'abaqus replay file not found'
    os.chdir(pyPath)
    #modelName = getPyScriptName()
    modelName = 'Model-1'
    mdb = Mdb(modelName+'.cae')
    myModel = mdb.Model(name=modelName)
    myAssembly = myModel.rootAssembly
    cliCommand("""session.journalOptions.setValues(replayGeometry=COORDINATE, recoverGeometry=COORDINATE)""")
    mappedMeshFlag = 1
    #------- CREATE GEOMETRY
    elem_size = 0.125
    totalsize = 1.0
    fac =1
    mesh_size = 0.02*fac
    plate_thickness = 1.0e-3
    numCells = totalsize/elem_size
    total_numCells = numCells**2
    #------- CREATE GEOMETRY: PLATES
    GeomName = 'Plate'
    partNames={}
    partNames[GeomName]=[]
    for i in range(int(total_numCells)):
        if i%8 == 0:
            m=i/8
            k = 0
        point2=((k+1)*elem_size, (m+1)*elem_size)
        point1=(k*elem_size, m*elem_size)
        k = k+1
        PlateName = GeomName +'-'+str(i+1)
        partNames[GeomName].append(PlateName)
        myModel.ConstrainedSketch(name='__profile__', sheetSize=5.0)
        myModel.sketches['__profile__'].rectangle(point1=point1,
            point2=point2)
        myModel.Part(dimensionality=TWO_D_PLANAR, name=PlateName, type=
            DEFORMABLE_BODY)
        myModel.parts[PlateName].BaseShell(sketch=
            myModel.sketches['__profile__'])
        del myModel.sketches['__profile__']
    #------ CREATE GEOMETRY: CRACK WIRE
    extra_size = 0.001
    crack_size = 0.2
    part_crack = 'initial_crack'
    myModel.ConstrainedSketch(name='__profile__', sheetSize=5.0)
    myModel.sketches['__profile__'].Line(point1=(0.0, totalsize/2+extra_size), point2=(
        crack_size, totalsize/2+extra_size))
    myModel.Part(dimensionality=TWO_D_PLANAR, name=part_crack, type=
        DEFORMABLE_BODY)
    myModel.parts[part_crack].BaseWire(sketch=myModel.sketches['__profile__'])
    del myModel.sketches['__profile__']
    #------- CREATE GEOMETRY: SET INSTANCES
    inst=[]
    for i in range(1,int(total_numCells)+1):
        PlateName = GeomName +'-'+str(i)
        print PlateName
        p = myModel.parts[PlateName]
        instanceName = PlateName
        currentSet=0
        myAttributes = p.queryAttributes()
        if(myAttributes['numShellFaces']==0 and myAttributes['numWireEdges']>0):
            currentSet = p.Set(name='Set_'+PlateName+str(j),edges=p.edges)
        else:
            currentSet = p.Set(name='Set_'+PlateName,faces=p.faces)
        myAssembly.Instance(name=instanceName,part=myModel.parts[PlateName],autoOffset=OFF,dependent=ON)
        inst.append(myAssembly.instances[instanceName])
    #------- CREATE GEOMETRY: SET CRACK WIRE
    instanceName = part_crack
    p = myModel.parts[part_crack]
    currentSet=0
    myAttributes = p.queryAttributes()
    if(myAttributes['numShellFaces']==0 and myAttributes['numWireEdges']>0):
        currentSet = p.Set(name='Set_'+part_crack,edges=p.edges)
    else:
        currentSet = p.Set(name='Set_'+part_crack,faces=p.faces)
    myAssembly.Instance(name=instanceName,part=myModel.parts[part_crack],autoOffset=OFF,dependent=ON)
    inst.append(myAssembly.instances[instanceName])
    #------ CREATE MATERIAL PROPERTY: SOFT MATERIAL
    vol = plate_thickness*totalsize
    E_SOFT = 100.0 # MPa
    NU_SOFT = 0.33
    SOFT_CRTICICAL_STRAIN = 0.4
    SIGMA_SOFT = E_SOFT*SOFT_CRTICICAL_STRAIN
    G_SOFT = 0.5*E_SOFT*SOFT_CRTICICAL_STRAIN*SOFT_CRTICICAL_STRAIN*vol
    myModel.Material(name='Soft_Material')
    myModel.materials['Soft_Material'].Elastic(dependencies=0,
        moduli=LONG_TERM, noCompression=OFF, noTension=OFF, table=((E_SOFT, NU_SOFT), )
        , temperatureDependency=OFF, type=ISOTROPIC)
    myModel.materials['Soft_Material'].MaxpsDamageInitiation(table=(
        (SIGMA_SOFT, ), ))
    myModel.materials['Soft_Material'].maxpsDamageInitiation.DamageEvolution(
        type=ENERGY, table=((G_SOFT, ), ))
    myModel.materials['Soft_Material'].maxpsDamageInitiation.DamageStabilizationCohesive(
        cohesiveCoeff=0.0005)
    myModel.materials['Soft_Material'].setValues(materialIdentifier='')
    #------ CREATE MATERIAL PROPERTY: STIFF MATERIAL
    E_STIFF = 1000.0 # MPa
    NU_STIFF = 0.33
    STIFF_CRTICICAL_STRAIN = 0.04
    SIGMA_STIFF = E_STIFF*STIFF_CRTICICAL_STRAIN
    G_STIFF = 0.5*E_STIFF*STIFF_CRTICICAL_STRAIN*STIFF_CRTICICAL_STRAIN*vol
    myModel.Material(name='Stiff_Material')
    myModel.materials['Stiff_Material'].Elastic(dependencies=0,
        moduli=LONG_TERM, noCompression=OFF, noTension=OFF, table=((E_STIFF, NU_STIFF),
        ), temperatureDependency=OFF, type=ISOTROPIC)
    myModel.materials['Stiff_Material'].MaxpsDamageInitiation(table=(
        (SIGMA_STIFF, ), ))
    myModel.materials['Stiff_Material'].maxpsDamageInitiation.DamageEvolution(
        type=ENERGY, table=((G_STIFF, ), ))
    myModel.materials['Stiff_Material'].maxpsDamageInitiation.DamageStabilizationCohesive(
        cohesiveCoeff=0.0005)
    myModel.materials['Stiff_Material'].setValues(materialIdentifier=
        '')
    #------ CREATE MATERIAL PROPERTY: CREATE SECTION
    myModel.HomogeneousSolidSection(material='Stiff_Material', name=
        'Section-Stiff', thickness=plate_thickness)
    myModel.HomogeneousSolidSection(material='Soft_Material', name=
        'Section-Soft', thickness=plate_thickness)
    #------ CREATE MATERIAL PROPERTY: ASSIGN SECTION PROPERTY
    '''
    ORIENTATION OF PLATES
    57 58 59 60 61 62 63 64
    49 50 51 52 53 54 55 56
    41 42 43 44 45 46 47 48
    33 34 35 36 37 38 39 40
    25 26 27 28 29 30 30 32
    17 18 19 20 21 22 23 24
    9 10 11 12 13 14 15 16
    1 2 3 4 5 6 7 8
    '''
    #------ CREATE MATERIAL PROPERTY: THIS IS WHERE THE SHUFFLING IS NEEDED BETWEEN STIFF AND SOFT
#     randlist=[]
#     randlist+=random.sample(range(1, 65), 32)
#     rlist=set(randlist)
#     for j in range(1, int(total_numCells)+1):
#         ii = j-1
#         if j in rlist:
# #           print myModel.parts[partNames[GeomName][j]].sets.keys()
#             region = myModel.parts[partNames[GeomName][ii]].sets['Set_'+partNames[GeomName][ii]]
#             myModel.parts[partNames[GeomName][ii]].SectionAssignment(region=region,
#             sectionName='Section-Soft', offset=0.0)
#         else:
# #           print myModel.parts[partNames[GeomName][j]].sets.keys()
#             region = myModel.parts[partNames[GeomName][ii]].sets['Set_'+partNames[GeomName][ii]]
#             myModel.parts[partNames[GeomName][ii]].SectionAssignment(region=region,
#             sectionName='Section-Stiff', offset=0.0)





    mesh = [25, 23, 40, 13, 44, 35, 26, 8, 28, 30, 6, 21, 27, 53, 29, 20, 17, 47, 55, 34, 31, 54, 58, 57, 36, 56, 46, 38, 45, 52, 48, 11]


    # mat55=[233, 218, 203, 188, 173, 158, 129, 131, 133, 135, 137, 139, 141, 142, 143, 144, 114, 116, 118, 120, 122, 124, 126, 127, 128, 110, 93, 76, 59, 42, 25, 8]
    total=[x for x in range(1,65)]
    for i in total:
        ii=i-1
        if i in mesh:
            region = myModel.parts[partNames[GeomName][ii]].sets['Set_'+partNames[GeomName][ii]]
            myModel.parts[partNames[GeomName][ii]].SectionAssignment(region=region,
            sectionName='Section-Soft', offset=0.0)
        else:
            region = myModel.parts[partNames[GeomName][ii]].sets['Set_'+partNames[GeomName][ii]]
            myModel.parts[partNames[GeomName][ii]].SectionAssignment(region=region,
            sectionName='Section-Stiff', offset=0.0)
    #------ CREATE ASSEMBLY
    myAssembly.DatumCsysByDefault(CARTESIAN)
    sizeCube = [totalsize,totalsize]
    #------ CREATE ASSEMBLY: CREATE SETS OF SURFACE
    PlateSurf = {}
    PlateSurfInstName = {}
    PlateSurf[GeomName]=[]
    PlateSurfInstName[GeomName]=[]
    for j in range(int(total_numCells)):
        myAssembly.Instance(name=partNames[GeomName][j],part=myModel.parts[partNames[GeomName][j]],autoOffset=OFF,dependent=ON)
    currentInclFaces=[]
    for j in range(int(total_numCells)):
        currentInst = myAssembly.instances[partNames[GeomName][j]]
        faces = currentInst.faces
        name='Surf_'+partNames[GeomName][j]
        surfEdges=[]
        surfEdgesId = faces[0].getEdges()
        for tmpId in surfEdgesId:
            if(not isEdgeExternal(currentInst,currentInst.edges[tmpId],sizeCube)):
                surfEdges.append(currentInst.edges[tmpId:tmpId+1])
        s = myAssembly.Surface(side1Edges=surfEdges,name=name)
        PlateSurf[GeomName].append(s)
        PlateSurfInstName[GeomName].append(currentInst.name)

    for i in range(1,int(total_numCells)):
        part_name = 'Part-'+str(i)
        next_plate = 'Plate-'+str(i+1)
        if i == 1:
            previous_part = 'Plate-'+str(i)
        else:
            previous_part = 'Part-'+str(i-1)+'-1'
        myAssembly.InstanceFromBooleanMerge(domain=GEOMETRY,
            instances=(myAssembly.instances[previous_part],
            myAssembly.instances[next_plate]),
            keepIntersections=ON, name=part_name, originalInstances=DELETE)
    allFaces=[]
    for inst in myAssembly.instances.values():
        faces=inst.faces
        allFaces.append(faces[0:len(faces)])
    setAllFaces = myAssembly.Set(name='allRVE',faces=allFaces)
    #------ CREATE ASSEMBLY: CREATE REFERENCE POINTS
    refpoint1 = myAssembly.ReferencePoint(point=(0.5*sizeCube[0],-0.1*sizeCube[1],0.0))
    refpoint2 = myAssembly.ReferencePoint(point=(0.5*sizeCube[0],1.1*sizeCube[1],0.0))
    myAssembly.Set(name='refpoint1',referencePoints=(myAssembly.referencePoints[refpoint1.id],))
    myAssembly.Set(name='refpoint2',referencePoints=(myAssembly.referencePoints[refpoint2.id],))
    myAssembly.Surface(name='Surf_Bottom', side1Edges=myAssembly.instances[part_name+'-1'].edges.findAt(
        ((0.96875, 0.0, 0.0), ), ((0.84375, 0.0, 0.0), ), ((0.71875, 0.0, 0.0), ),
        ((0.59375, 0.0, 0.0), ), ((0.46875, 0.0, 0.0), ), ((0.34375, 0.0, 0.0), ),
        ((0.21875, 0.0, 0.0), ), ((0.09375, 0.0, 0.0), ), ))
    myAssembly.Surface(name='Surf_Top', side1Edges=myAssembly.instances[part_name+'-1'].edges.findAt(
        ((0.90625, 1.0, 0.0), ), ((0.78125, 1.0, 0.0), ), ((0.65625, 1.0, 0.0), ),
        ((0.53125, 1.0, 0.0), ), ((0.40625, 1.0, 0.0), ), ((0.28125, 1.0, 0.0), ),
        ((0.15625, 1.0, 0.0), ), ((0.03125, 1.0, 0.0), ), ))
#    myAssembly.Surface(name='Surf_HalfTop', side1Edges=myAssembly.instances[part_name+'-1'].edges.findAt(
#    ((0.28125, 1.0, 0.0), ), ((0.15625, 1.0, 0.0), ), ((0.03125, 1.0, 0.0), )))
    #------ CREATE STEP
    myModel.StaticStep(adaptiveDampingRatio=0.05,continueDampingFactors=False,
    timePeriod=2.0, initialInc=1e-05, maxNumInc=10000000, minInc=1e-10, maxInc=2.0, name='Step-1',
    nlgeom=OFF, previous='Initial', stabilizationMagnitude=0.0002,
    stabilizationMethod=DISSIPATED_ENERGY_FRACTION)
    #------ CREATE STEP: CHANGE F-OUTPUT
    myModel.fieldOutputRequests['F-Output-1'].setValues(
        numIntervals=100, variables=('S', 'PE', 'PEEQ', 'PEMAG', 'LE', 'U', 'RF',
        'CF', 'PHILSM', 'PSILSM', 'STATUSXFEM', 'IVOL', 'EVOL'))
    #------ CREATE STEP: CHANGE SOLVER CONTROLS
    myModel.steps['Step-1'].control.setValues(
        allowPropagation=OFF, resetDefaultValues=OFF, timeIncrementation=(10.0,
        20.0, 9.0, 16.0, 10.0, 4.0, 12.0, 10.0, 6.0, 3.0, 50.0))
    #------ CREATE INTERACTIONS: COUPLINGS
    myModel.Coupling(controlPoint=myAssembly.sets['refpoint1'],
        couplingType=KINEMATIC, influenceRadius=WHOLE_SURFACE, localCsys=None,
        name='Coupling-1', surface=myAssembly.surfaces['Surf_Bottom'], u1=ON, u2=ON, ur3=ON)
    myModel.Coupling(controlPoint=myAssembly.sets['refpoint2'],
        couplingType=KINEMATIC, influenceRadius=WHOLE_SURFACE, localCsys=None,
        name='Coupling-2', surface=myAssembly.surfaces['Surf_Top'], u1=ON, u2=ON, ur3=ON)
    #------ CREATE INTERACTIONS: XFEM
    myAssembly.engineeringFeatures.XFEMCrack(crackDomain=myAssembly.sets['allRVE'],
        crackLocation=myAssembly.instances[part_crack].sets['Set_'+part_crack]
        , name='Crack-1')
    #------ CREATE BOUNDARY CONDITIONS: FIXED AT BOTTOM FACES
    myModel.DisplacementBC(amplitude=UNSET, createStepName='Step-1', distributionType=UNIFORM,
        fieldName='', fixed=OFF, localCsys=None, name='BC-1',
        region=myAssembly.sets['refpoint1'], u1=0.0, u2=0.0, ur3=0.0)
    #------ CREATE BOUNDARY CONDITIONS: LOADING AT TOP FACES
    Srate = 0.2 # STRAIN RATE
    myModel.EquallySpacedAmplitude(begin=0.0, data=(0.0, 1.0), fixedInterval=1.0, name='Amp-1',
        smooth=SOLVER_DEFAULT, timeSpan=STEP)
    myModel.DisplacementBC(amplitude='Amp-1', createStepName='Step-1', distributionType=UNIFORM,
        fieldName='', fixed=OFF, localCsys=None, name='BC-2',
        region=myAssembly.sets['refpoint2'], u1=0.0, u2=sizeCube[0]*Srate, ur3=0.0)
    myAssembly.regenerate()
    #------ CREATE BOUNDARY CONDITIONS: ROLLER AT SIDE FACES
    e1 = myAssembly.instances['Part-63-1'].edges
    edges1 = e1.findAt(((1.0, 0.96875, 0.0), ), ((0.0, 0.90625, 0.0), ), ((1.0,
        0.84375, 0.0), ), ((0.0, 0.78125, 0.0), ), ((1.0, 0.71875, 0.0), ), ((
        0.0, 0.65625, 0.0), ), ((1.0, 0.59375, 0.0), ), ((0.0, 0.53125, 0.0),
        ), ((1.0, 0.46875, 0.0), ), ((0.0, 0.40625, 0.0), ), ((1.0, 0.34375,
        0.0), ), ((0.0, 0.28125, 0.0), ), ((1.0, 0.21875, 0.0), ), ((0.0,
        0.15625, 0.0), ), ((1.0, 0.09375, 0.0), ), ((0.0, 0.03125, 0.0), ))
    myAssembly.Set(edges=edges1, name='Set_LRedges')
    region = myAssembly.sets['Set_LRedges']
    myModel.DisplacementBC(name='BC-3',
        createStepName='Step-1', region=region, u1=0.0, u2=UNSET, ur3=0.0,
        amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='',
        localCsys=None)
    #------ CREATE MESHES
    deviationFactor=0.07
    minSizeFactor=0.1
    remeshFlag = NO
    myModel.parts[part_name].seedPart(size=mesh_size,
        deviationFactor=deviationFactor, minSizeFactor=minSizeFactor)
    #------ CREATE MESHES: ELEMENT SHAPES (TRI, QUAD, QUAD_DOMINATED)
    f1 = myModel.parts[part_name].faces
    faces1 = f1.findAt(((0.916667,
        0.916667, 0.0), ), ((0.791667, 0.916667, 0.0), ), ((0.666667, 0.916667,
        0.0), ), ((0.541667, 0.916667, 0.0), ), ((0.416667, 0.916667, 0.0), ), ((
        0.291667, 0.916667, 0.0), ), ((0.166667, 0.916667, 0.0), ), ((0.041667,
        0.916667, 0.0), ), ((0.916667, 0.791667, 0.0), ), ((0.791667, 0.791667,
        0.0), ), ((0.666667, 0.791667, 0.0), ), ((0.541667, 0.791667, 0.0), ), ((
        0.416667, 0.791667, 0.0), ), ((0.291667, 0.791667, 0.0), ), ((0.166667,
        0.791667, 0.0), ), ((0.041667, 0.791667, 0.0), ), ((0.916667, 0.666667,
        0.0), ), ((0.791667, 0.666667, 0.0), ), ((0.666667, 0.666667, 0.0), ), ((
        0.541667, 0.666667, 0.0), ), ((0.416667, 0.666667, 0.0), ), ((0.291667,
        0.666667, 0.0), ), ((0.166667, 0.666667, 0.0), ), ((0.041667, 0.666667,
        0.0), ), ((0.916667, 0.541667, 0.0), ), ((0.791667, 0.541667, 0.0), ), ((
        0.666667, 0.541667, 0.0), ), ((0.541667, 0.541667, 0.0), ), ((0.416667,
        0.541667, 0.0), ), ((0.291667, 0.541667, 0.0), ), ((0.166667, 0.541667,
        0.0), ), ((0.041667, 0.541667, 0.0), ), ((0.916667, 0.416667, 0.0), ), ((
        0.791667, 0.416667, 0.0), ), ((0.666667, 0.416667, 0.0), ), ((0.541667,
        0.416667, 0.0), ), ((0.416667, 0.416667, 0.0), ), ((0.291667, 0.416667,
        0.0), ), ((0.166667, 0.416667, 0.0), ), ((0.041667, 0.416667, 0.0), ), ((
        0.916667, 0.291667, 0.0), ), ((0.791667, 0.291667, 0.0), ), ((0.666667,
        0.291667, 0.0), ), ((0.541667, 0.291667, 0.0), ), ((0.416667, 0.291667,
        0.0), ), ((0.291667, 0.291667, 0.0), ), ((0.166667, 0.291667, 0.0), ), ((
        0.041667, 0.291667, 0.0), ), ((0.916667, 0.166667, 0.0), ), ((0.791667,
        0.166667, 0.0), ), ((0.666667, 0.166667, 0.0), ), ((0.541667, 0.166667,
        0.0), ), ((0.416667, 0.166667, 0.0), ), ((0.291667, 0.166667, 0.0), ), ((
        0.166667, 0.166667, 0.0), ), ((0.041667, 0.166667, 0.0), ), ((0.916667,
        0.041667, 0.0), ), ((0.791667, 0.041667, 0.0), ), ((0.666667, 0.041667,
        0.0), ), ((0.541667, 0.041667, 0.0), ), ((0.416667, 0.041667, 0.0), ), ((
        0.291667, 0.041667, 0.0), ), ((0.166667, 0.041667, 0.0), ), ((0.041667,
        0.041667, 0.0), ), )
    myModel.parts[part_name].setMeshControls(algorithm=MEDIAL_AXIS, elemShape=QUAD, regions=faces1)
    #------ CREATE MESHES: ELEMENT TYPES(PlaneStress = CPS4R, PlaneStrain = CPE4R)
    myModel.parts[part_name].setElementType(elemTypes=(ElemType(elemCode=CPS4R, elemLibrary=STANDARD,
        secondOrderAccuracy=OFF, hourglassControl=ENHANCED, distortionControl=DEFAULT, elemDeletion=ON),
        ElemType(elemCode=CPE4R, elemLibrary=STANDARD)), regions=(faces1, ))
    myModel.parts[part_name].generateMesh()
    #------ CREATE JOB
    job_name = '50Soft_'+getPyScriptName()
    mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF, explicitPrecision=SINGLE,
    getMemoryFromAnalysis=True, historyPrint=OFF, memory=90, memoryUnits=PERCENTAGE, model=modelName,
    modelPrint=OFF, multiprocessingMode=DEFAULT, name=job_name, nodalOutputPrecision=SINGLE, numCpus=6,
    numDomains=6, numGPUs=0, queue=None, resultsFormat=ODB,
    scratch='', type=ANALYSIS, userSubroutine='', waitHours=0, waitMinutes=0)
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)

if __name__=='__main__':
    main()
