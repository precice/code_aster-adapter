import os
import sys
import numpy as np
import numpy.linalg
from mpi4py import MPI

##	Not sure about the following import, it used to be 
##	from Cata.cata import *
##	from Utilitai.partition import *

from code_aster.Cata.Syntax import *
from code_aster.Cata.DataStructure import *
from code_aster.Cata.Commons import *

np.set_printoptions(threshold=np.inf)  #Makes sure that Numpy won't summarise arrays, but fully represents them

##	Define preCICE and adapter location

precice_root = os.getenc('PRECICE_ROOT')	#Change this to a proper representation
precice_python_adapter_root = precice_root+"/src/precice/adapters/python"
sys.path.insert(0, precice_python_adapter_root)		#Changes the active path to the adapter one

##	Check the following import, it used to be:
##	import PySolverInterfacett
##	from PySolverInterface import *

from precice import *
from constants import *

class Adapter:
	
	def __init__(self, preciceConfigFile, participantName, config, MESH, MODEL, MAT, isNonLinear=False):
        self.interfaces = []
        self.numInterfaces = len(config)
        self.MESH = MESH
        self.MODEL = MODEL
        self.MAT = MAT
        self.LOADS = []
        self.isNonLinear = isNonLinear
        self.participantName = participantName
        self.preciceDt = -1
		self.precice = precice.Interface(paricipantName, 0, 1)
        self.configure(config)
		
	def configure(self, config):
		L = [None] * self.numInterfaces 	## Loads
		SM = [None] * self.numInterfaces	## Shifted meshes
		for i in range(self.numInterfaces):
			SM[i] = CREA_MAILLAGE(MAILLAGE=self.MESH, RESTREINT={"GROUP_MA": config[i]["patch"], "GROUP_NO": config[i]["patch"]})
			##	self.precice is not yet defines in __init__
			##	I deleted self.precice as an argument from Interface()
			##	I suspect that it's not necessary with precice.pyx
			interface = Interface(self.precice, self.participantName, config[i], self.MESH, self.MODEL, self.MAT[config[i]["material-id"]], self.isNonLinear)
			BCs = interface.setLoad(L[i])	##	Check where setLoad comes from
			L[i] = AFFE_CHAR_THER(MODELE=self.MODEL, ECHANGE=BCs)
			interface.setLoad(L[i])
			self.LOADS.append({'CHARGE': L[i]})
			self.interfaces.append(interface)
			
	def initialize(self, INIT_T): #cleared
		
		self.preciceDt = self.precice.initialize()

        if self.precice.is_action_required(constants.action_write_initial_data()):
            self.writeCouplingData(INIT_T)
            self.precice.fulfilled_action(constants.action_write_initial_data())

        self.precice.initialize_data()

        return self.preciceDt
	
	def isCouplingOngoing(self): #cleared
		return self.precice.is_coupling_ongoing()
		
	def writeCouplingData(self, TEMP):
		if self.precice.is_write_data_required(self.preciceDt):
			for interface in self.interfaces:
				interface.writesBCs(TEMP)
		
	def readCouplingData(): code #cleared
		if self.precice.is_read_data_available():
			for interface in self.interfaces:
				interface.readAndupdateBCs() ##	readAndupdateBCs() comes from the interface class below
		
	def writeCheckpoint(): code #cleared
		if self.precice.is_action_required(constants.action_write_iteration_checkpoint()):
			# Do nothing
			self.precice.fulfilled_action(constants.action_write_iteration_checkpoint())
		
	def readCheckpoint(): #cleared
		if self.precice.is_action_required(constants.action_read_iteration_checkpoint()):
			# Do nothing
			self.precice.fulfilled_action(constants.action_read_iteration_checkpoint())			
		
	def isCouplingTimestepComplete(): #cleared
		return self.precice.is_timestep_complete()
		
	def advance(): #cleared
		self.preciceDt = self.precice.advance(self.preciceDt)
		return self.preciceDt
		
	def finalize(): #cleared
		self.precice.finalize()
		
		
	
class Interface:
	##	I deleted the precice parameter
    def __init__(self, precice, participantName, names, MESH, SHMESH, MODEL, MAT, isNonLinear = False):

		self.precice = precice
        self.participantName = participantName

        self.groupName = ""
        self.facesMeshName = ""
        self.nodesMeshName = ""

        self.nodes = []
        self.faces = []
        self.connectivity = []
        self.nodeCoordinates = []
        self.faceCenterCoordinates = []
        self.normals = None

        self.isNonLinear = isNonLinear
        self.conductivity = None
        self.isConductivityInitialized = False
        self.delta = 1e-5

        self.preciceNodeIndices = []
        self.preciceFaceCenterIndices = []

        self.preciceFaceCentersMeshID = 0
        self.preciceNodesMeshID = 0

        self.readHCoeffDataID = 0
        self.readTempDataID = 0
        self.writeHCoeffDataID = 0
        self.writeTempDataID = 0

        self.readTemp = []
        self.readHCoeff = []
        self.writeTemp = []
        self.writeHCoeff = []

        self.readDataSize = 0
        self.writeDataSize = 0

        self.MESH = MESH
        # Shifted mesh (contains only the interface, and is shifted by delta in the direction opposite to the normal)
        self.SHMESH = SHMESH
        self.MODEL = MODEL
        self.MAT = MAT
        self.LOAD = None
        self.LOADS = []
        self.mesh = MAIL_PY()			## This must be some Code_Aster specific code
        self.mesh.FromAster(MESH)		## This must be some Code_Aster specific code

        self.configure(names)
    
    def configure(self, names):

        self.groupName = names["patch"]

        # In Code_Aster, write-data is located at the nodes
        self.nodesMeshName = names["write-mesh"]
        # and read-data is located at the face centers
        self.faceCentersMeshName = names["read-mesh"]
        
        self.computeNormals()
        
        self.nodeCoordinates = np.array([p for p in self.SHMESH.sdj.COORDO.VALE.get()])
        self.nodeCoordinates = np.resize(self.nodeCoordinates, (len(self.nodeCoordinates)/3, 3))
        self.shiftMesh()
        
        self.faces = [self.mesh.correspondance_mailles[idx] for idx in self.mesh.gma[self.groupName]]
        self.connectivity = [self.mesh.co[idx] for idx in self.mesh.gma[self.groupName]]
        self.faceCenterCoordinates = np.array([np.array([self.mesh.cn[node] for node in elem]).mean(0) for elem in self.connectivity])

        self.setVertices()

        self.setDataIDs(names)
        
        self.readDataSize = len(self.faces)
        self.writeDataSize = len(self.nodeCoordinates)
        
        self.readHCoeff = np.zeros(self.readDataSize)
        self.readTemp = np.zeros(self.readDataSize)
		
	def computeNormals(self):
		# Get normals at the nodes
        DUMMY = AFFE_MODELE(
            MAILLAGE=self.SHMESH,
            AFFE={
                'TOUT': 'OUI',
                'PHENOMENE': 'THERMIQUE',
                'MODELISATION': '3D',
            },
        )
        N = CREA_CHAMP(
            MODELE=DUMMY,
            TYPE_CHAM='NOEU_GEOM_R',
            GROUP_MA=self.groupName,
            OPERATION='NORMALE'
        )
        self.normals = N.EXTR_COMP().valeurs
        self.normals = np.resize(np.array(self.normals), (len(self.normals)/3, 3))
        DETRUIRE(CONCEPT=({"NOM": N}, {"NOM": DUMMY}))
		
	def setVertices(self):
		# Nodes
        self.preciceNodeIndices = [0] * len(self.nodeCoordinates)
        self.preciceNodesMeshID = self.precice.get_mesh_ID(self.nodesMeshName)
        self.precice.set_mesh_vertices(self.preciceNodesMeshID, len(self.nodeCoordinates), np.hstack(self.nodeCoordinates), self.preciceNodeIndices)
        # Face centers
        self.preciceFaceCenterIndices = [0] * len(self.faceCenterCoordinates)
        self.preciceFaceCentersMeshID = self.precice.get_mesh_id(self.faceCentersMeshName)
        self.precice.set_mesh_vertices(self.preciceFaceCentersMeshID, len(self.faceCenterCoordinates), np.hstack(self.faceCenterCoordinates), self.preciceFaceCenterIndices)
        
	def setDataIDs(self, names):
		for writeDataName in names["write-data"]:
            if writeDataName.find("Heat-Transfer-Coefficient-") >= 0: 	## why bigger or equal to zero?
                self.writeHCoeffDataID = self.precice.get_data_id(writeDataName, self.preciceNodesMeshID)
            elif writeDataName.find("Sink-Temperature-") >= 0:
                self.writeTempDataID = self.precice.get_data_id(writeDataName, self.preciceNodesMeshID)
        for readDataName in names["read-data"]:
            if readDataName.find("Heat-Transfer-Coefficient-") >= 0:
                self.readHCoeffDataID = self.precice.get_data_id(readDataName, self.preciceFaceCentersMeshID)
            elif readDataName.find("Sink-Temperature-") >= 0:
                self.readTempDataID = self.precice.get_data_id(readDataName, self.preciceFaceCentersMeshID)
		
	def getPreciceNodeIndices(self):
		return self.preciceNodeIndices
		
	def getPreciceFaceCenterIndices(self):
		return self.preciceFaceCenterIndices
		
	def getPreciceNodesMeshID(self):
		return self. preciceNodesMeshID
		
	def getPreciceFaceCentersMeshID(self):
		return self.preciceFaceCentersMeshID
		
	def getNodeCoordinates(self):
		return self.nodeCoordinates
		
	def getFaceCenterCoordinates(self):
		return self.faceCenterCoordinates
		
	def getNormals(self):
		return self.normals
		
	def createBSs(self):
		"""
        Note: TEMP_EXT and COEF_H need to be initialized with different values, otherwise Code_Aster
        will group identical values together, and it will not be possible to apply different BCs
        to different element faces.  Additionally, COEF_H must be different from 0
        (otherwise it will be grouped with a default internal 0).
        """
        BCs = [
            {'MAILLE': self.faces[j], 'TEMP_EXT': j, 'COEF_H': j+1}
            for j in range(len(self.faces))
        ]
        return BCs
		
	def updateBCs(self, temp, hCoeff):		##	I don't really get what self.LOAD.sdj means here
		
		self.LOAD.sdj.CHTH.T_EXT.VALE.changeJeveuxValues(len(temp),
                                        tuple(np.array(range(len(temp))) * 10 + 1),
                                        tuple(temp),
                                        tuple(temp),
                                        1)
        self.LOAD.sdj.CHTH.COEFH.VALE.changeJeveuxValues(len(hCoeff),
                                        tuple(np.array(range(1, len(hCoeff)+1)) * 3 + 1),
                                        tuple(hCoeff),
                                        tuple(hCoeff),
                                        1)
		
	def readAndUpdateBCs(self):
		self.precice.read_block_scalar_data(self.readHCoeffDataID, self.readDataSize, self.preciceFaceCenterIndices, self.readHCoeff)
        self.precice.read_block_scalar_data(self.readTempDataID, self.readDataSize, self.preciceFaceCenterIndices, self.readTemp)
        self.updateBCs(self.readTemp, self.readHCoeff)
		
	def writeBCs(self, TEMP):
		writeTemp, writeHCoeff = self.getBoundaryValues(TEMP)
        self.precice.write_block_scalar_data(self.writeHCoeffDataID, self.writeDataSize, self.preciceNodeIndices, writeHCoeff)
        self.precice.write_block_scalar_data(self.writeTempDataID, self.writeDataSize, self.preciceNodeIndices, writeTemp)

	def getBoundaryValues(self, T):
		
		# Sink temperature
        TPROJ = PROJ_CHAMP(MAILLAGE_1=self.MESH, MAILLAGE_2=self.SHMESH, CHAM_GD=T, METHODE='COLLOCATION')
        writeTemp = TPROJ.EXTR_COMP(lgno=[self.groupName]).valeurs
        DETRUIRE(CONCEPT=({'NOM': TPROJ}))

        # Heat transfer coefficient
        self.updateConductivity(writeTemp)
        writeHCoeff = np.array(self.conductivity) / self.delta

        return writeTemp, writeHCoeff
		
	def setLoad(self, LOAD):
		self.LOAD = LOAD
		
	def shiftMesh(self):
		coords = [p for p in self.SHMESH.sdj.COORDO.VALE.get()]
        for i in range(len(self.normals)):
            for c in range(3):
                coords[i*3 + c] = coords[i*3 + c] - self.normals[i][c] * self.delta
        self.SHMESH.sdj.COORDO.VALE.changeJeveuxValues(len(coords), tuple(range(1, len(coords)+1)), tuple(coords), tuple(coords), 1)

	def updateConductivity(self, T):
		if self.isNonLinear:
            self.conductivity = [self.MAT.RCVALE("THER_NL", nompar="TEMP", valpar=t, nomres="LAMBDA")[0][0] for t in T]
            self.isConductivityInitialized = True
        elif not self.isConductivityInitialized:
            self.conductivity = [self.MAT.RCVALE("THER", nompar="TEMP", valpar=t, nomres="LAMBDA")[0][0] for t in T]
            self.isConductivityInitialized = True
        # Note: RCVALE returns ((LAMBDA,),(0,)), therefore we use [0][0] to extract the value of LAMBDA

	