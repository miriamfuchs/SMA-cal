#! /sma/local/anaconda/bin/python
#/usr/bin/env python
#/opt/cfpython/anaconda2.7/bin/python
# This is a test comment

import numpy as np
import scipy.io
import os, sys, struct, mmap, argparse, shutil
from astropy.time import Time
from math import sin, cos, pi, sqrt, factorial
import csv

verbose = True
fullRxList  = [0, 1, 2, 3]
fullSbList  = [0, 1]
fullWinList = [0, 1, 2, 3]

def normalize0to360(angle):
    while angle >= 360.0:
        angle -= 360.0
    while angle < 0.0:
        angle += 360.0
    return angle

def toGeocentric(X, Y, Z):
    """
    Transform antenna coordinates to FITS-IDI style GEOCENTRIC coordinates
    """
    SMALong = -2.713594675620429
    SMALat  = 0.3459976585365961
    SMARad  = 6382.248*1000.0
    Rx = SMARad*cos(SMALong)*cos(SMALat)
    Ry = SMARad*sin(SMALong)*cos(SMALat)
    Rz = SMARad*sin(SMALat)
    x = X*cos(SMALong) - Y*sin(SMALong) #+ Rx
    y = Y*cos(SMALong) + X*sin(SMALong) #+ Ry
    z = Z #+ Rz
    return (x, y, z)

def makeInt(data, size):
    tInt = 0
    for i in range(size):
        tInt += ord(data[i])<<(i<<3)
    return tInt

def makeFloat(data):
    return (struct.unpack('f', data[:4]))[0]

def makeDouble(data):
    return (struct.unpack('d', data[:8]))[0]

def lookupTsys(tsysMap,tsysOffset):
    nVals = makeInt(tsysMap[tsysOffset:tsysOffset+4],4)
    tsysVal = 0.0
    for idx in range(nVals):
        newOffset = tsysOffset + 12 + (idx*16)
        tsysVal = max(tsysVal,makeFloat(tsysMap[newOffset:newOffset+4]))
    return tsysVal

def lookupVis(visMap,visOffset,nChan,realOnly=False,imagOnly=False):
    rawSpec = np.frombuffer(visMap[visOffset:visOffset+(4*nChan)+2],dtype=np.int16)
    scaleFac = np.double(2.0**rawSpec[0])
    if realOnly:
        dataSpec = rawSpec[1:(2*nChan)+1:2]
    elif imagOnly:
        dataSpec = rawSpec[2:(2*nChan)+1:2]
    else:
        dataSpec = rawSpec[1:(2*nChan)+1]
    return (dataSpec,scaleFac)

def blStructSubset(blStruct,blScreen):
    subBlStruct = {
        'baselineID' : blStruct['baselineID'][blScreen],
        'intID'      : blStruct['intID'][blScreen],
        'sidebandID' : blStruct['sidebandID'][blScreen],
        'polID'      : blStruct['polID'][blScreen],
        'ant1RxID'   : blStruct['ant1RxID'][blScreen],
        'ant2RxID'   : blStruct['ant2RxID'][blScreen],
        'receiverID' : blStruct['receiverID'][blScreen],
        'uCoords'    : blStruct['uCoords'][blScreen],
        'vCoords'    : blStruct['vCoords'][blScreen],
        'wCoords'    : blStruct['wCoords'][blScreen],
        'midTime'    : blStruct['midTime'][blScreen],
        'ant1ID'     : blStruct['ant1ID'][blScreen],
        'ant2ID'     : blStruct['ant2ID'][blScreen],
        'ant1Tsys'   : blStruct['ant1Tsys'][blScreen],
        'ant2Tsys'   : blStruct['ant2Tsys'][blScreen]
    }
    return subBlStruct

def spStructSubset(spStruct,spScreen):
    subSpStruct = {
        'specwinID'   : spStruct['specwinID'][spScreen],
        'baselineID'  : spStruct['baselineID'][spScreen],
        'intID'       : spStruct['intID'][spScreen],
        'winID'       : spStruct['winID'][spScreen],
        'sidebandID'  : spStruct['sidebandID'][spScreen],
        'receiverID'  : spStruct['receiverID'][spScreen],
        'gainCal'     : spStruct['gainCal'][spScreen],
        'passbandCal' : spStruct['passbandCal'][spScreen],
        'restFreq'    : spStruct['restFreq'][spScreen],
        'freqCenter'  : spStruct['freqCenter'][spScreen],
        'freqRes'     : spStruct['freqRes'][spScreen],
        'gunnLO'      : spStruct['gunnLO'][spScreen],
        'cabinLO'     : spStruct['cabinLO'][spScreen],
        'corrLO1'     : spStruct['corrLO1'][spScreen],
        'corrLO2'     : spStruct['corrLO2'][spScreen],
        'nChan'       : spStruct['nChan'][spScreen],
        'dataOffset'  : spStruct['dataOffset'][spScreen],
        'flagVals'    : spStruct['flagVals'][spScreen],
        'scaleFac'    : spStruct['scaleFac'][spScreen],
        'normFac'    : spStruct['normFac'][spScreen],
    }
    return subSpStruct


parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description='Convert an SMA MIR format dataset into a set of MATLAB-readable files.')
parser.add_argument('dataset',           help='The data directory to process (i.e. /sma/rtdata/...)')
parser.add_argument('destination',       help='The directory to write out data (e.g. /sma/reduction/...)')
args = parser.parse_args()
dataDir = args.dataset
destDir = args.destination

if not os.path.exists(destDir):
    os.mkdir(destDir)

##################################
# Get antenna position values
antXYZ = np.zeros([sum(1 for line in open(dataDir+'/antennas')), 4]);
count = 0
for line in open(dataDir+'/antennas','r'):
    tok = line.split()
    antNum = float(tok[0])
    antXPos = float(tok[1])
    antYPos = float(tok[2])
    antZPos = float(tok[3])
    antXYZ[count] = np.append(np.asarray(toGeocentric(antXPos, antYPos, antZPos)),antNum)
    count += 1
# Finish reading ant positions
##################################

##################################
# Begin reading in codes_read
fileCodes = open(dataDir+'/codes_read', 'rb')
data = fileCodes.read()
codesRecLen = 42
nCodesRecords = len(data)/codesRecLen
# First make a list with all unique code strings
codesDict = {}
codeStrings = []
for rec in range(nCodesRecords):
    codeString = ''
    for i in range(12):
        if data[codesRecLen*rec + i] < ' ':
            break
        if (data[codesRecLen*rec + i] >= ' ') and (data[codesRecLen*rec + i] <= 'z'):
            codeString += data[codesRecLen*rec + i]
    icode = makeInt(data[codesRecLen*rec + 12:], 2)
    payload = ''
    for i in range(14, 14+26):
        if data[codesRecLen*rec + i] < ' ':
            break
        if (data[codesRecLen*rec + i] >= ' ') and (data[codesRecLen*rec + i] <= 'z'):
            payload += data[codesRecLen*rec + i]
    if codeString not in codeStrings:
        codeStrings.append(codeString)
        codesDict[codeString] = {}
    codesDict[codeString][icode] = payload

fileCodes.close()
gainCodes = codesDict['gq'].copy()
gainCodes[0] = (gainCodes[0] == 'g')
gainCodes[1] = (gainCodes[1] == 'g')

passCodes = codesDict['pq'].copy()
passCodes[0] = (passCodes[0] == 'p')
passCodes[1] = (passCodes[1] == 'p')

winCode = {'c1' : np.int32(-1), 's1' : np.int32(0), 's2' : np.int32(1), 's3': np.int32(2),
           's4': np.int32(3), 's5': np.int32(4), 's6': np.int32(5), 's7': np.int32(6), 's8': np.int32(7)}
specbandCode = codesDict['band'].copy()
for winID in specbandCode.keys():
    specbandCode[winID] = winCode[specbandCode[winID]]

sbDict = {'l': 0, 'u': 1}
sbCodes = codesDict['sb'].copy()
for sbKey in sbCodes.keys():
    sbCodes[sbKey] = sbDict[sbCodes[sbKey]]

monDict =   {'Jan':  1, 'Feb':  2, 'Mar':  3, 'Apr':  4, 'May':  5, 'Jun':  6,
             'Jul':  2, 'Aug':  8, 'Sep':  9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

mjdCodes = codesDict['ut'].copy()

for mjdKey in mjdCodes.keys():
    try:
        dateString = mjdCodes[mjdKey]
        yearVal = int(dateString[7:11])
        monthVal =  monDict[dateString[0:3]]
        dayVal = int(dateString[4:6])
        hourVal = int(dateString[12:14]) + 12*(dateString[24:26] == 'PM')
        remTimeVal = dateString[14:24]
        mjdCodes[mjdKey] = Time("%0004i-%02i-%02i %02i%s" % (yearVal, monthVal, dayVal, hourVal, remTimeVal),scale='utc').mjd
    except ValueError:
        mjdCodes[mjdKey] = 0.0

souCodes = codesDict['source'].copy()

# Finish with codes structure
###########################################

##################################
# Begin dealing with per-integration headers
if verbose:
    print 'Reading in_read'

fileIn = open(dataDir+'/in_read', 'rb')
data = fileIn.read()
inRecLen = 188
nInRecords = len(data)/inRecLen
trialID = np.zeros(nInRecords,dtype=np.int32)
intID = np.zeros(nInRecords,dtype=np.int32)
azVal = np.zeros(nInRecords,dtype=np.double)
elVal = np.zeros(nInRecords,dtype=np.double)
intTime = np.zeros(nInRecords,dtype=np.double)
mjdTime = np.zeros(nInRecords,dtype=np.double)
projectID = np.zeros(nInRecords,dtype=np.int32)
souName = np.zeros(nInRecords,dtype=np.object)
souID = np.zeros(nInRecords,dtype=np.int32)
#souCode = np.zeros(nInRecords,dtype=np.int32)
offsetXDec = np.zeros(nInRecords,dtype=np.double)
offsetDec = np.zeros(nInRecords,dtype=np.double)
raCode = np.zeros(nInRecords,dtype=np.int32)
decCode = np.zeros(nInRecords,dtype=np.int32)
raSou = np.zeros(nInRecords,dtype=np.double)
decSou = np.zeros(nInRecords,dtype=np.double)
inDict = {}
for rec in range(nInRecords):
    trialID[rec]    = makeInt(data[rec*inRecLen +   0:], 4)
    intID[rec]      = makeInt(data[rec*inRecLen +   4:], 4)
    azVal[rec]      = makeFloat(data[rec*inRecLen +  12:])
    elVal[rec]      = makeFloat(data[rec*inRecLen +  16:])
    intTime[rec]    = makeFloat(data[rec*inRecLen +  64:])
    mjdTime[rec]    = mjdCodes[intID[rec]]
    projectID[rec]  = makeInt(data[rec*inRecLen +  68:], 4) # project id
    #souID[rec]      = makeInt(data[rec*inRecLen +  72:], 4) # sou id
    souID[rec]      = makeInt(data[rec*inRecLen +  76:], 2) # sou code ID
    souName[rec]    = souCodes[souID[rec]]
    offsetXDec[rec] = makeFloat(data[rec*inRecLen +  80:])
    offsetDec[rec]  = makeFloat(data[rec*inRecLen +  84:])
    raCode[rec]     = makeInt(data[rec*inRecLen +  88:], 2)
    decCode[rec]    = makeInt(data[rec*inRecLen +  90:], 2)
    raSou[rec]      = makeDouble(data[rec*inRecLen +  92:])
    decSou[rec]     = makeDouble(data[rec*inRecLen + 100:])
    inDict[intID[rec]] = rec

fileIn.close()
inStruct = {
    'trialID'    : trialID,
    'intID'      : intID,
    'azVal'      : azVal,
    'elVal'      : elVal,
    'intTime'    : intTime,
    'mjdTime'    : mjdTime,
    'projectID'  : projectID,
    'souID'      : souID,
    'souName'    : souName,
    'offsetXDec' : offsetXDec,
    'offsetDec'  : offsetDec,
    'raCode'     : raCode,
    'decCode'    : decCode,
    'raSou'      : raSou,
    'decSou'     : decSou
}

# Finish with integration structure
################################################

######################################
# Begin reading in per-antenna headers
if verbose:
    print 'Reading in antenna data...'

fileEng = open(dataDir+'/eng_read', 'rb')
engRecLen = 196
fSize = os.path.getsize(dataDir+'/eng_read')
nEngRecords = fSize/engRecLen
antNumber = np.zeros(nEngRecords,dtype=np.int32)
padNumber = np.zeros(nEngRecords,dtype=np.int32)
antStatus = np.zeros(nEngRecords,dtype=np.int32)
intID     = np.zeros(nEngRecords,dtype=np.int32)
hrAngle   = np.zeros(nEngRecords,dtype=np.double)
lstTime   = np.zeros(nEngRecords,dtype=np.double)
azErr     = np.zeros(nEngRecords,dtype=np.double)
elErr     = np.zeros(nEngRecords,dtype=np.double)
tsysRx1   = np.zeros(nEngRecords,dtype=np.double)
tsysRx2   = np.zeros(nEngRecords,dtype=np.double)

for rec in range(nEngRecords):
    if ((rec % 10000) == 0) and verbose:
        print '\t processing record %d of %d (%2.0f%% done)' % (rec, nEngRecords, 100.0*float(rec)/float(nEngRecords))
        sys.stdout.flush()
    data = fileEng.read(engRecLen)
    antNumber[rec] =    makeInt(data[  0:], 4)
    padNumber[rec] =    makeInt(data[  4:], 4)
    antStatus[rec] =    makeInt(data[  8:], 4)
    intID[rec]     =    makeInt(data[ 20:], 4)
    hrAngle[rec]   = makeDouble(data[ 36:])
    lstTime[rec]   = makeDouble(data[ 44:])
    azErr[rec]     = makeDouble(data[116:])
    elErr[rec]     = makeDouble(data[124:])
    tsysRx1[rec]   = makeDouble(data[172:])
    tsysRx2[rec]   = makeDouble(data[180:])

fileEng.close()
engStruct = {
    'antNumber' : antNumber,
    'padNumber' : padNumber,
    'antStatus' : antStatus,
    'intID'     : intID,
    'hrAngle'   : hrAngle,
    'lstTime'   : lstTime,
    'azErr'     : azErr,
    'elErr'     : elErr,
    'tsysRx1'   : tsysRx1,
    'tsysRx2'   : tsysRx2
}
# Finish with antenna structure
################################################
 
######################################
# Begin reading in per-baseline headers
if verbose:
    print 'Reading in baseline data...'

fileBl = open(dataDir+'/bl_read', 'rb')
fSize = os.path.getsize(dataDir+'/bl_read')
blRecLen = 158
nBlRecords = fSize/blRecLen
baselineID = np.zeros(nBlRecords,dtype=np.int32)
intID = np.zeros(nBlRecords,dtype=np.int32)
sidebandID = np.zeros(nBlRecords,dtype=np.int32)
polID = np.zeros(nBlRecords,dtype=np.int32)
ant1RxID = np.zeros(nBlRecords,dtype=np.int32)
ant2RxID = np.zeros(nBlRecords,dtype=np.int32)
pntStatus = np.zeros(nBlRecords,dtype=np.int32)
receiverID = np.zeros(nBlRecords,dtype=np.int32)
uCoords = np.zeros(nBlRecords,dtype=np.double)
vCoords = np.zeros(nBlRecords,dtype=np.double)
wCoords = np.zeros(nBlRecords,dtype=np.double)
midTime = np.zeros(nBlRecords,dtype=np.double)
ant1ID = np.zeros(nBlRecords,dtype=np.int32)
ant2ID = np.zeros(nBlRecords,dtype=np.int32)
ant1Tsys = np.zeros(nBlRecords,dtype=np.double)
ant2Tsys = np.zeros(nBlRecords,dtype=np.double)
blDict = {}

# Open up access to tsys values
tsysFile = os.open(dataDir+'/tsys_read', os.O_RDONLY)
tsysMap = mmap.mmap(tsysFile, 0, prot=mmap.PROT_READ);

for rec in range(nBlRecords):
    if ((rec % 10000) == 0) and verbose:
        print '\t processing record %d of %d (%2.0f%% done)' % (rec, nBlRecords, 100.0*float(rec)/float(nBlRecords))
        sys.stdout.flush()
    data = fileBl.read(blRecLen)
    if len(data) == blRecLen:
        baselineID[rec] =  makeInt(data[  0:], 4)
        intID[rec]      =  makeInt(data[  4:], 4)
        sidebandID[rec] =  sbCodes[makeInt(data[  8:], 2)]
        polID[rec]      =  makeInt(data[ 10:], 2)
        ant1RxID[rec]   =  makeInt(data[ 12:], 2)
        ant2RxID[rec]   =  makeInt(data[ 14:], 2)
        pntStatus[rec]  =  makeInt(data[ 16:], 2)
        receiverID[rec] =  makeInt(data[ 18:], 2)
        uCoords[rec]    =  makeFloat(data[ 20:])
        vCoords[rec]    =  makeFloat(data[ 24:])
        wCoords[rec]    =  makeFloat(data[ 28:])
        midTime[rec]    =  makeDouble(data[ 40:])
        ant1ID[rec]     =  makeInt(data[ 60:], 2)
        ant2ID[rec]     =  makeInt(data[ 62:], 2)
        ant1TsysOff =  makeInt(data[ 64:], 4)
        ant1Tsys[rec]   = lookupTsys(tsysMap, ant1TsysOff)
        ant2TsysOff =  makeInt(data[ 68:], 4)
        ant2Tsys[rec]   = lookupTsys(tsysMap, ant2TsysOff)
        blDict[baselineID[rec]] = rec

tsysMap.close()
os.close(tsysFile)
fileBl.close()
blStruct = {
    'baselineID' : baselineID,
    'intID'      : intID,
    'sidebandID' : sidebandID,
    'polID'      : polID,
    'ant1RxID'   : ant1RxID,
    'ant2RxID'   : ant2RxID,
    'receiverID' : receiverID,
    'uCoords'    : uCoords,
    'vCoords'    : vCoords,
    'wCoords'    : wCoords,
    'midTime'    : midTime,
    'ant1ID'     : ant1ID,
    'ant2ID'     : ant2ID,
    'ant1Tsys'   : ant1Tsys,
    'ant2Tsys'   : ant2Tsys
}

# Finish with BL structure
##################################

##############################################
# Begin reading per spectral window headers
if verbose:
    print 'Reading individual band metadata...'

fileSp = open(dataDir+'/sp_read', 'rb')
fSize = os.path.getsize(dataDir+'/sp_read')
spRecLen = 188
nSpRecords = fSize/spRecLen

specwinID = np.zeros(nSpRecords,dtype=np.int32)
baselineID = np.zeros(nSpRecords,dtype=np.int32)
intID = np.zeros(nSpRecords,dtype=np.int32)
gainCal = np.zeros(nSpRecords,dtype=np.bool)
passbandCal = np.zeros(nSpRecords,dtype=np.bool)
winID = np.zeros(nSpRecords,dtype=np.int32)
sidebandID = np.zeros(nSpRecords,dtype=np.int32)
receiverID = np.zeros(nSpRecords,dtype=np.int32)
freqCenter = np.zeros(nSpRecords,dtype=np.double)
freqRes = np.zeros(nSpRecords,dtype=np.double)
gunnLO = np.zeros(nSpRecords,dtype=np.double)
cabinLO = np.zeros(nSpRecords,dtype=np.double)
corrLO1 = np.zeros(nSpRecords,dtype=np.double)
corrLO2 = np.zeros(nSpRecords,dtype=np.double)
polCode = np.zeros(nSpRecords,dtype=np.int32)
nChan =  np.zeros(nSpRecords,dtype=np.int32)
dataOffset = np.zeros(nSpRecords,dtype=np.int32)
flagVals =  np.zeros(nSpRecords,dtype=np.int32)
restFreq =  np.zeros(nSpRecords,dtype=np.double)
scaleFac =  np.zeros(nSpRecords,dtype=np.double)
normFac =  np.zeros(nSpRecords,dtype=np.double)
spDict = {}

for rec in range(nSpRecords):
    if ((rec % 100000) == 0) and verbose:
        print '\t processing record %d of %d (%2.0f%% done)' % (rec, nSpRecords, 100.0*float(rec)/float(nSpRecords))
        sys.stdout.flush()
    data = fileSp.read(spRecLen)
    if len(data) == spRecLen:
        specwinID[rec]   = makeInt(data[  0:], 4)
        baselineID[rec]  = makeInt(data[  4:], 4)
        intID[rec]       = makeInt(data[  8:], 4)
        gainCal[rec]     = gainCodes[makeInt(data[ 12:], 2)]
        passbandCal[rec] = passCodes[makeInt(data[ 14:], 2)]
        winID[rec]       = specbandCode[makeInt(data[ 16:], 2)]
        sidebandID[rec]  = blStruct['sidebandID'][blDict[baselineID[rec]]]
        receiverID[rec]  = blStruct['receiverID'][blDict[baselineID[rec]]]
        freqCenter[rec]  = makeDouble(data[ 36:])
        freqRes[rec]     = makeFloat(data[ 44:])
        gunnLO[rec]      = makeDouble(data[ 48:])
        cabinLO[rec]     = makeDouble(data[ 56:])
        corrLO1[rec]     = makeDouble(data[ 64:])
        corrLO2[rec]     = makeDouble(data[ 72:])
        flagVals[rec]    = makeInt(data[ 88:], 4)
        nChan[rec]       = makeInt(data[ 96:], 2)
        dataOffset[rec]  = makeInt(data[100:], 4)
        restFreq[rec]    = makeDouble(data[104:])
        spDict[specwinID[rec]] = rec

fileSp.close()

spStruct = {
    'specwinID'   : specwinID,
    'baselineID'  : baselineID,
    'intID'       : intID,
    'winID'       : winID,
    'sidebandID'  : sidebandID,
    'receiverID'  : receiverID,
    'gainCal'     : gainCal,
    'passbandCal' : passbandCal,
    'restFreq'    : restFreq,
    'freqCenter'  : freqCenter,
    'freqRes'     : freqRes,
    'gunnLO'      : gunnLO,
    'cabinLO'     : cabinLO,
    'corrLO1'     : corrLO1,
    'corrLO2'     : corrLO2,
    'nChan'       : nChan,
    'dataOffset'  : dataOffset,
    'flagVals'    : flagVals,
    'scaleFac'    : scaleFac,
    'normFac'     : normFac
}
# Finish SP structure
#####################

#####################
# Begin reading in autocorrelation data
if verbose:
    print 'Reading autocorrelation data...'

fileAuto = open(dataDir+'/autoCorrelations', 'rb')
recordFile = destDir + '/auto'
fSize = os.path.getsize(dataDir+'/autoCorrelations')
autoRecLen = 20+(4*2*16384*4)
nAutoRecords = fSize/autoRecLen

if os.path.exists(destDir + '/autow0p0.bin'):
    os.remove(destDir + '/autow0p0.bin')
if os.path.exists(destDir + '/autow0p1.bin'):
    os.remove(destDir + '/autow0p1.bin')

if os.path.exists(destDir + '/autow1p0.bin'):
    os.remove(destDir + '/autow1p0.bin')
if os.path.exists(destDir + '/autow1p1.bin'):
    os.remove(destDir + '/autow1p1.bin')

if os.path.exists(destDir + '/autow2p0.bin'):
    os.remove(destDir + '/autow2p0.bin')
if os.path.exists(destDir + '/autow2p1.bin'):
    os.remove(destDir + '/autow2p1.bin')

if os.path.exists(destDir + '/autow3p0.bin'):
    os.remove(destDir + '/autow3p0.bin')
if os.path.exists(destDir + '/autow3p1.bin'):
    os.remove(destDir + '/autow3p1.bin')

antID = np.zeros(nAutoRecords,dtype=np.int32)
#nChunks = np.zeros(nAutoRecords,dtype=np.int32)
intID = np.zeros(nAutoRecords,dtype=np.int32)
dHrs =  np.zeros(nAutoRecords,dtype=np.float)

#with open(recordFile,"a") as recFile:
for rec in range(nAutoRecords):
    if ((rec % 1000) == 0) and verbose:
        print '\t processing record %d of %d (%2.0f%% done)' % (rec, nAutoRecords, 100.0*float(rec)/float(nAutoRecords))
    sys.stdout.flush()
    data = fileAuto.read(autoRecLen)
    antID[rec] = makeInt(data[0:4], 4)
    #nChunks[rec] = makeInt(data[4:8],4)
    intID[rec] = makeInt(data[8:12], 4)
    dHrs[rec] = makeDouble(data[12:20])
    autoSpec = np.frombuffer(data[20:],dtype=np.float32)
    
    # Do the first polarization
    recFile = open(recordFile + 'w0p0.bin','a')
    recFile.write(autoSpec[(0l*16384):(1l*16384)].tobytes())
    recFile.close()
    recFile = open(recordFile + 'w0p1.bin','a')
    recFile.write(autoSpec[(1l*16384):(2l*16384)].tobytes())
    recFile.close()
    recFile = open(recordFile + 'w1p0.bin','a')
    recFile.write(autoSpec[(2l*16384):(3l*16384)].tobytes())
    recFile.close()
    recFile = open(recordFile + 'w1p1.bin','a')
    recFile.write(autoSpec[(3l*16384):(4l*16384)].tobytes())
    recFile.close()
    recFile = open(recordFile + 'w2p0.bin','a')
    recFile.write(autoSpec[(4l*16384):(5l*16384)].tobytes())
    recFile.close()
    recFile = open(recordFile + 'w2p1.bin','a')
    recFile.write(autoSpec[(5l*16384):(6l*16384)].tobytes())
    recFile.close()
    recFile = open(recordFile + 'w3p0.bin','a')
    recFile.write(autoSpec[(6l*16384):(7l*16384)].tobytes())
    recFile.close()
    recFile = open(recordFile + 'w3p1.bin','a')
    recFile.write(autoSpec[(7l*16384):(8l*16384)].tobytes())
    recFile.close()

fileAuto.close()

autoStruct = {
    'antID'   : antID,
    'intID'  : intID,
    'dHrs'    : dHrs
}
# Finish AC structure
#####################


visFile = os.open(dataDir+'/sch_read', os.O_RDONLY)
visMap = mmap.mmap(visFile, 0, prot=mmap.PROT_READ)

dataOffset = 0
inOffsetDict = {}
while dataOffset < len(visMap):
    intID = makeInt(visMap[dataOffset:dataOffset+4],4)
    if ((intID % 100) == 0) and verbose:
        sys.stdout.write("Pre-processing integration records (%2.0f%% done)   \r" % (100.0*float(intID)/float(nInRecords)))
        sys.stdout.flush()
    inOffsetDict[intID] = dataOffset+8
    dataOffset += makeInt(visMap[dataOffset+4:dataOffset+8], 4) + 8
sys.stdout.write("Integration pre-processing complete!                                        \n")
sys.stdout.flush()
visMap.close()
os.close(visFile)

winList = np.unique(spStruct['winID'])
winScreen = np.zeros(winList.shape,dtype=bool)
sbList = np.unique(spStruct['sidebandID'])
sbScreen = np.zeros(sbList.shape,dtype=bool)
rxList = np.unique(spStruct['receiverID'])
rxScreen = np.zeros(rxList.shape,dtype=bool)

for winID in range(len(winList)):
    if winList[winID] in fullWinList:
        winScreen[winID] = True

for sbID in range(len(sbList)):
    if sbList[sbID] in fullSbList:
        sbScreen[sbID] = True

for rxID in range(len(rxList)):
    if rxList[rxID] in fullRxList:
        rxScreen[rxID] = True

winList = winList[winScreen]
sbList = sbList[sbScreen]
rxList = rxList[rxScreen]

marker = 0
"""
for rxID in rxList:
    for sbID in sbList:
        for winID in winList:
            sys.stdout.write("Writing out visibilities (%2.3g%% complete)   \r" % (100.0*marker/(len(winList)*len(sbList)*len(rxList))))
            sys.stdout.flush()
            visFile = os.open(dataDir+'/sch_read', os.O_RDONLY)
            visMap = mmap.mmap(visFile, 0, prot=mmap.PROT_READ);
            subDir = "RX%1d-SB%1d-WN%02d" % (rxID, sbID, winID)
            subDirPath = destDir + '/' + subDir
            if os.path.exists(subDirPath):
                if os.path.exists(subDirPath + '/real.bin'):
                    os.remove(subDirPath + '/real.bin')
                if os.path.exists(subDirPath + '/imag.bin'):
                    os.remove(subDirPath + '/imag.bin')
                if os.path.exists(subDirPath + '/vis.bin'):
                    os.remove(subDirPath + '/vis.bin')
                if os.path.exists(subDirPath + '/spdata.mat'):
                    os.remove(subDirPath + '/spdata.mat')
                if os.path.exists(subDirPath + '/bldata.mat'):
                    os.remove(subDirPath + '/bldata.mat')
                if os.path.exists(subDirPath + '/endata.mat'):
                    os.remove(subDirPath + '/endata.mat')
                if os.path.exists(subDirPath + '/indata.mat'):
                    os.remove(subDirPath + '/indata.mat')             
                if os.path.exists(subDirPath + '/indata.mat'):
                    os.remove(subDirPath + '/indata.mat')             
                if os.path.exists(subDirPath + '/indata.mat'):
                    os.remove(subDirPath + '/indata.mat')             
                if os.path.exists(subDirPath + '/indata.mat'):
                    os.remove(subDirPath + '/indata.mat')             
                if os.path.exists(subDirPath + '/indata.mat'):
                    os.remove(subDirPath + '/indata.mat')
                if os.path.exists(subDirPath + '/autodata.mat'):
                    os.remove(subDirPath + '/autodata.mat')
                if os.path.exists(subDirPath + '/auto.bin'):
                    os.remove(subDirPath + '/auto.bin')
            else:
                os.mkdir(subDirPath)
            blScreen = np.logical_and(blStruct['receiverID'] == rxID, blStruct['sidebandID'] == sbID)
            spScreen = np.logical_and(np.logical_and(spStruct['receiverID'] == rxID, spStruct['sidebandID'] == sbID), spStruct['winID'] == winID)
            typicalNChan = np.median(spStruct['nChan'][spScreen])
            spScreen = np.logical_and(spScreen, spStruct['nChan'] == typicalNChan)
            recordFile = subDirPath + '/vis.bin'
            with open(recordFile,"a") as recFile:
                for indvRec in np.where(spScreen)[0]:
                    intID = spStruct['intID'][indvRec]
                    blRecID = blDict[spStruct['baselineID'][indvRec]]
                    normFac = sqrt(2.0/pi)*130.0*np.sqrt(np.abs(blStruct['ant1Tsys'][blRecID]*blStruct['ant2Tsys'][blRecID]))
                    dataOffset = spStruct['dataOffset'][indvRec] + inOffsetDict[intID]
                    rawVis, scaleFac = lookupVis(visMap, dataOffset, spStruct['nChan'][indvRec])
                    recFile.write(rawVis.tobytes())
                    spStruct['scaleFac'][indvRec] = scaleFac
                    spStruct['normFac'][indvRec] = normFac
            subBlStruct = blStructSubset(blStruct, blScreen)
            subSpStruct = spStructSubset(spStruct, spScreen)
            scipy.io.savemat(subDirPath + '/spdata.mat',subSpStruct,oned_as='row')
            scipy.io.savemat(subDirPath + '/bldata.mat',subBlStruct,oned_as='row')
            scipy.io.savemat(subDirPath + '/endata.mat',engStruct,oned_as='row')
            scipy.io.savemat(subDirPath + '/indata.mat',inStruct,oned_as='row')
            scipy.io.savemat(subDirPath + '/antxyz.mat',{'antXYZ' : antXYZ},oned_as='row')
            scipy.io.savemat(subDirPath + '/autodata.mat',autoStruct,oned_as='row')
            autoFile = "autow%1dp%1d.bin" % (winID, rxID/2)
            shutil.copy(destDir + '/' + autoFile, subDirPath + '/auto.bin')
            visMap.close()
            os.close(visFile)
            marker += 1

sys.stdout.write("Writing out visibilities complete!                     \n")
sys.stdout.flush()

# Clean up autocorrelation data
os.remove(destDir + '/autow0p0.bin')
os.remove(destDir + '/autow0p1.bin')
os.remove(destDir + '/autow1p0.bin')
os.remove(destDir + '/autow1p1.bin')
os.remove(destDir + '/autow2p0.bin')
os.remove(destDir + '/autow2p1.bin')
os.remove(destDir + '/autow3p0.bin')
os.remove(destDir + '/autow3p1.bin')
"""
#want to save invidual files for the following:

np.save('spStructFile', spStruct) #info for each individ chunk
np.save('BlStructFile', blStruct) #info for baseline (per rx per sideband per baseline per scan)
np.save('engStructFile', engStruct) #info on fixed length records (for each ant after each scan completes)
np.save('inStructFile', inStruct) #info on scan header, per each scan


#-------calibrator sources lists-------------
#
gainSources={'0004-476','0005+383','0006-063','0006+243','0010+109','0013+408','0014+612','0019+203','0019+260','0019+734','0022+002','0034+279','0050-094','0051-068','0057+303','0102+584','0106-405','0108+015','0112+227','0113+498','0115-014','0116-116','0118-216','0120-270','0121+118','0121+043','0132-169','0136+478','0137-245','0141-094','0149+059','0149+189','0152+221','0204+152','0204-170','0205+322','0210-510','0217+738','0217+017','0219+013','0222-346','0224+069','0228+673','0237+288','0238+166','0239+042','0241-082','0242+110','0242-215','0244+624','0246-468','0251+432','0259+425','0303+472','0309+104','0310+382','0313+413','0319+415','0325+469','0325+226','0329-239','0334-401','0336+323','0339-017','0340-213','0346+540','0348-278','0354+467','0359+509','0359+323','0401+042','0403-360','0405-131','0406-384','0410+769','0415+448','0416-209','0418+380','0422+023','0423-013','0423+418','0424-379','0424+006','0428+329','0428-379','0433+053','0440-435','0442-002','0449+113','0449+635','0453-281','0455-462','0457-234','0457+067','0501-019','0502+061','0502+136','0502+416','0505+049','0509+056','0510+180','0512+152','0522-364','0526-485','0527+035','0530+135','0532+075','0533+483','0538-440','0539+145','0539-286','0541-056','0541+474','0542+498','0555+398','0605+405','0607-085','0608-223','0609-157','0625+146','0629-199','0646+448','0648-307','0650-166','0710+475','0717+456','0721+713''0725+144','0725-009','0730-116','0733+503','0738+177','0739+016','0741+312','0747+766','0747-331','0748+240','0750+482','0750+125','0753+538','0757+099','0802+181','0804-278','0806-268','0808-078','0808+498','0811+017','0818+423','0823+223','0824+558','0824+392','0825+031','0826-225','0828-375','0830+241','0831+044','0836-202','0840+132','0841+708','0854+201','0902-142','0903+468','0909+013','0914+027','0920+446','0921+622','0925+003','0927+390','0927-205','0937+501','0943-083','0948+406','0956+252','0957+553','0958+474','0958+655','1008+063','1010+828','1014+230','1018+357','1033+608','1035-201','1037-295','1039-156','1041+061','1043+241','1044+809','1048-191','1048+717','1048+717','1051+213','1057-245','1058+812','1058+015','1102+279','1103+302','1103+220','1104+382','1107-448','1107+164','1111+199','1112-219','1118+125','1120-251','1120+143','1122+180','1127-189','1130-148','1130-148','1145-228','1146-289','1146+539','1146+399','1147-382','1153+809','1153+495','1159-224','1159+292','1203+480','1205-265','1207+279','1209-241','1215-175','1218-460','1221+282','1222+042','1224+213','1227+365','1229+020','1230+123','1239+075','1244+408','1246-075','1246-257','1248-199','1254+116','1256-057','1258-223','1305-105','1309+119','1310+323','1316-336','1325-430','1327+221','1329+319','1331+305','1337-129','1337-129','1357-154','1357+767','1408-078','1415+133','1416+347','1419+543','1419+383','1424-492','1427-421','1433-158','1439-155','1446+173','1454-377','1454-402','1458+042','1459+716','1504+104','1505+034','1506+426','1507-168','1510-057','1512-090','1513-102','1513-212','1516+002','1517-243','1522-275','1538+003','1540+147','1549+506','1549+026','1549+026','1550+054','1553+129','1557-000','1604-446','1608+104','1613+342','1617+027','1619+227','1625-254','1626-298','1632+825','1635+381','1637+472','1638+573','1640+397','1642+689','1642+398','1653+397','1658+347','1658+476','1658+076','1700-261','1707+018','1713-269','1716+686','1719+177','1727+455','1728+044','1733-130','1734+389','1734+094','1737+063','1739+499','1740+521','1743-038','1744-312','1751+096','1753+288','1800+388','1800+784','1801+440','1802-396','1806+698','1821+397','1824+568','1829+487','1830+063','1832-206','1842+681','1848+323','1848+327','1849+670','1902+319','1911-201','1923-210','1924+156','1924-292','1925+211','1927+612','1927+739','1937-399','1955+515','1957-387','2000-178','2005+778','2007+404','2009-485','2009+724','2011-157','2012+464','2015+371','2016+165','2023+318','2025+032','2025+337','mwc349a','2035+109','2038+513','2047-166','2049+100','2050+041','2056-472','2057-375','2101-295','2101+036','2109+355','2123+055','2129-183','2131-121','2134-018','2134-018','2138-246','2139+143','2142-046','2147+094','2148+069','2151+071','2151-304','2152+175','2158-150','2158-302','2202+422','2203+317','2203+174','2206-185','2213-254','2217+243','2218-035','2225-049','2229-085','2232+117','2235-485','2236+284','2243-257','2246-121','2247+031','2248-325','2253+161','2258-279','2301+374','2302-373','2307+148','2311+344','2320+052','2321+275','2323-032','2327+096','2329-475','2331-159','2333-237','2334+076','2337-025','2347+517','2347-512','2348-165','2354+458','2358-103',' 3c454.3','3c446',' bllac','p2134+0','3c418','3c395','3c380','3c371','nrao530','mrk501','3c345','nrao512','3c309.1','3c286','cen a','3c279',' 3c274','3c273','mrk421','oj287','3c207','3c147',' 3c120','3c111','3c84','ngc315','3C454.3','3C446',' bllac','p2134+0','3C418','3C395','3C380','3C371','nrao530','mrk501','3C345','nrao512','3C309.1','3C286','Cen a','3C279',' 3C274','mrk421','oj287','3C273','3C207','3C147','3C120','3C111','3C84'}

fluxSources={'callisto', 'Callisto', 'ganymede', 'Ganymede', 'uranus', 'Uranus', 'Jupiter', 'jupiter', 'venus', 'Venus', 'mars', 'Mars', 'titan', 'Titan', 'saturn', 'Saturn', 'neptune', 'Neptune'}
bandpassSources={'3c454.3', '3C454.3', '3c279', '3C279', '3C273', '3c273', '3c84', '3C84'}


containsGain = [x in gainSources for x in inStruct['souName']] #returns TRUE if scans belong to a gain calibrator
containsFlux = [y in fluxSources for y in inStruct['souName']] #returns TRUE if scans belong to flux calibrator
containsBandpass = [z in bandpassSources for z in inStruct['souName']] #returns TRUE if scans belong to a bandpass calibrator

#want to be able to select specific scans that belong to the above groups using boolean mask
selectGain = np.where(containsGain)
selectFlux = np.where(containsFlux)
selectBandpass = np.where(containsBandpass)

gainScans = inStruct['intID'][selectGain]
fluxScans = inStruct['intID'][selectFlux]
bandpassScans = inStruct['intID'][selectBandpass]

#create a boolean mask based on intID for other structures such as blStruct
blStructGain = [i in gainScans for i in blStruct['intID']] #poorly named; will fix later



#----------Aug updates---------
#in_read is fixed length records, one record per scan
#eng_read is fixed length records, one reocrd per antenna after each scan completes 
#bl_read is fixed length records, one record per rx per sideband per baseline per scan
#sp_read is fixed length records, one per chunk per rx per baseline per scan



#Need to pull out flagged data
#flagged_baseline = np.zeros(len(blStruct['baselineID']))
flagged_baseline=[]
flagged_scan=[]
flagged_ant1 = []
flagged_ant2 = []
flagged_rxid = []
flagged_baselineID = []
for l in range(0,len(blStruct['baselineID'])):
    print(l)
    baseID_mask=np.where(blStruct['baselineID'][l]==spStruct['baselineID'])
    if np.any(spStruct['flagVals'][baseID_mask]!=0):
          flagged_baseline.append(1)  #if 1 then flagged
          flagged_scan.append(blStruct['intID'][l]) #add scan ID 
          flagged_ant1.append(blStruct['ant1ID'][l])
          flagged_ant2.append(blStruct['ant2ID'][l])
          flagged_rxid.append(blStruct['receiverID'][l])
          flagged_baselineID.append(blStruct['baselineID'][l])
#Test code for tsys calc
numScans=len(inStruct['intID'])
sigma_int_rx1 = [] #create empty list to store calculated tsys values for EACH scan
sigma_int_rx2 = [] 

boltzmann = 1.38e-23 #Boltzmann constant [m^2 kg s^-2 K^-1]
freq_bandwidth=8e9
A_eff = 28.3
aper_eff = .8
calList = set()

#flagged_scans = np.zeros(len(flagged_baseline))
flagStruct = {} #set up blank dict to fill in after calculations
flagStruct['Scan_ID']=np.zeros(len(flagged_baseline))
flagStruct['Flagged']=np.zeros(len(flagged_baseline))
flagStruct['Ant1ID']=np.zeros(len(flagged_baseline))
flagStruct['Ant2ID']=np.zeros(len(flagged_baseline))
flagStruct['baselineID']=np.zeros(len(flagged_baseline))

#fill in flagStruct
flagStruct['Scan_ID'] = flagged_scan
flagStruct['Flagged'] = flagged_baseline
flagStruct['Ant1ID'] = flagged_ant1
flagStruct['Ant2ID'] = flagged_ant2
flagStruct['baseline_ID']=flagged_baselineID

for k in range(0,numScans): 
    if not (containsGain[k] or containsBandpass[k]):
        sigma_int_rx1.append(0)
        sigma_int_rx2.append(0)
        continue
    sigma_baselines_rx1 = [] #create empty list to store sigma values for each baseline for each scan (typically 28)
    sigma_baselines_rx2 = [] 
 
    scan_mask=np.where(engStruct['intID']==inStruct['intID'][k]) #should only select the 8 or so scans with same int ID
    flag_mask=np.where(blStruct['intID']==inStruct['intID'][k])
  
    print(k, scan_mask)
    int_time=inStruct['intTime'][k]
    N_antennas=len(engStruct['intID'][scan_mask]) #typically 8 but depends on project
    calList.add(inStruct['souName'][k])

    for l in range(0, N_antennas):
        for j in range(l+1, N_antennas): 
            #added a try/exception in order to see where crazy tsys values are coming from (can include if statement later if necessary)
            try:
                single_baseline_tsys_rx1= sqrt(engStruct['tsysRx1'][scan_mask][l]*engStruct['tsysRx1'][scan_mask][j])
                single_baseline_tsys_rx2= sqrt(engStruct['tsysRx2'][scan_mask][l]*engStruct['tsysRx2'][scan_mask][j])
            except ValueError:
                print('rx1',k,l,j,engStruct['tsysRx2'][scan_mask][l],engStruct['tsysRx2'][scan_mask][j])
                print('rx2',k,l,j,engStruct['tsysRx2'][scan_mask][l],engStruct['tsysRx2'][scan_mask][j])
           # single_baseline_tsys_rx2=round(single_baseline_tsys,4) 
            sigma_baselines_rx1.append(1e26*2*boltzmann*single_baseline_tsys_rx1/(A_eff*aper_eff*sqrt(freq_bandwidth*int_time)))              
            sigma_baselines_rx2.append(1e26*2*boltzmann*single_baseline_tsys_rx2/(A_eff*aper_eff*sqrt(freq_bandwidth*int_time)))
    
    sigma_int_rx1.append((sum([(x**-2) for x in sigma_baselines_rx1]))**-0.5)#get tsys for each scan from baselines (vs single ant)
    #tsys_int_array_rx1=np.array(tsys_int)#turn into array                      
    sigma_int_rx2.append((sum([(x**-2) for x in sigma_baselines_rx2]))**-0.5)#get tsys for each scan from baselines (vs single ant)
    

  


gunnLORx1=np.median(spStruct['gunnLO'][np.where(np.logical_or(spStruct['receiverID']==0,spStruct['receiverID']==2))])
gunnLORx2=np.median(spStruct['gunnLO'][np.where(np.logical_or(spStruct['receiverID']==1,spStruct['receiverID']==3))])
calList=list(calList)



     


#make blank array to fill in with tsys measurement for each rx for each scan
#numScans = len(engStruct['antNumber'])/8 #should be equiv to # of scans from Scan Id in other structures

sigmaStruct = {} #set up blank dict to fill in after calculations
sigmaStruct['Avg_sigma_rx1'] = np.zeros(len(calList),dtype=np.float)
sigmaStruct['Avg_sigma_rx2'] = np.zeros(len(calList),dtype=np.float)
sigmaStruct['Gunn_LO_rx1']=np.zeros(len(calList),dtype=np.float)
sigmaStruct['Gunn_LO_rx2']=np.zeros(len(calList),dtype=np.float)
sigmaStruct['souName'] = np.zeros(len(calList),dtype=object)


#sigma calc for each sources (vs each scan)
for h in range(0,len(calList)):
    source_match = np.where(calList[h]==inStruct['souName'])
    print(len(source_match))
    #source_match=np.array(source_match)
    #need to convert sigma_int lists in arrays to be able to index them with source_match
    sigma_int_rx1=np.array(sigma_int_rx1)
    sigma_int_rx2=np.array(sigma_int_rx2)
  
    sigmaStruct['souName'][h]= calList[h]
    sigmaStruct['Avg_sigma_rx1'][h]=sum([(z**-2) for z in sigma_int_rx1[source_match]])**-0.5
    sigmaStruct['Avg_sigma_rx2'][h]=sum([(z**-2) for z in sigma_int_rx2[source_match]])**-0.5
    sigmaStruct['Gunn_LO_rx1'][h] = gunnLORx1 
    sigmaStruct['Gunn_LO_rx2'][h] = gunnLORx2 

  
def out_csv(sigStruct, fileName):
    n = len(sigStruct['souName'])
    records = []
    for i in range(n):
        record = (sigStruct['souName'][i], 
                  sigStruct['Avg_sigma_rx1'][i],
                  sigStruct['Avg_sigma_rx2'][i],
                  sigStruct['Gunn_LO_rx1'][i],
                  sigStruct['Gunn_LO_rx2'][i])
        records.append(record)
    
    with open(fileName, "wt") as f:
        f.write('{}, {}, {}, {}, {}\n'.format('souName', 'Avg_sigma_rx1','Avg_sigma_rx2','Gunn_LO_rx1','Gunn_LO_rx2'))
        for rec in records:
            f.write('{}, {}, {}, {}, {}\n'.format(*rec)) #* allows unpacking of tuple without iterating over all elements

outdir = sys.argv[2]
outfile = os.path.basename(sys.argv[1].strip("/")) + ".csv"
outpath = os.path.join(outdir, outfile)
#outdir changed to outpath
 
out_csv(sigmaStruct, outpath)   





'''
sigmaStruct['Avg_sigma_Rx1'][intID - 1] = sigma_nu
# sigmaStruct['Avg_sigma_Rx2'][intID - 1] = meanSigma(tsysRx2, antStatus, N_baselines)
sigmaStruct['Scan_ID'][intID - 1] =  inStruct['intID'] 
sigmaStruct['baseline_count'] = N_baselines
'''


#----- Mimi Notes -------------
#to stay working within same python environment after launching script
#import sys
#sys.argv=['sma2matlab.py','/sma/data/science/mir_data/170319_06:12:06/','test1']
#execfile('sma2matlab.py')
#/sma/data/science/mir_data/170319_06:12:06

#/sma/data/flux/mir_data/170803_03:14:13/' --> file with weird rx codes 


#inStruct.keys()
#inStruct['SouName'] ...can always pare down with [] (choosing specific key)

#want to go thru science data tracks, need to know a couple things 1) is this a calib, if so, then save! if not, throw away (boolean data)...next is 2) is it a planet? if so, save! ..planet, known passband, normal calibrator
 

