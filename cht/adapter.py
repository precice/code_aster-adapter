import os
import sys
import numpy as np
import numpy.linalg
#from mpi4py import MPI
from code_aster.Cata.Commands import *
from Utilitai import partition
import precice

np.set_printoptions(threshold=np.inf)

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
		self.precice = precice.Participant(participantName, preciceConfigFile, 0, 1)
		self.configure(config)

	def configure(self, config):
		L = [None] * self.numInterfaces 	# Loads
		SM = [None] * self.numInterfaces	# Shifted meshes
		for i in range(self.numInterfaces):
			# Shifted mesh (interface nodes displaced by a distance delta in the direction of the surface normal
			SM[i] = CREA_MAILLAGE(MAILLAGE=self.MESH, RESTREINT={"GROUP_MA": config[i]["patch"], "GROUP_NO": config[i]["patch"]})
			# Create interface
			interface = Interface(self.precice, self.participantName, config[i], self.MESH, SM[i], self.MODEL, self.MAT[config[i]["material-id"]], self.isNonLinear)
			# Loads
			BCs = interface.createBCs()
			L[i] = AFFE_CHAR_THER(MODELE=self.MODEL, ECHANGE=BCs)
			interface.setLoad(L[i])
			self.LOADS.append({'CHARGE': L[i]})
			self.interfaces.append(interface)

	def initialize(self, INIT_T):
		if self.precice.requires_initial_data():
			self.writeCouplingData(INIT_T)
		self.precice.initialize()

	def isCouplingOngoing(self):
		return self.precice.is_coupling_ongoing()

	def writeCouplingData(self, TEMP):
		for interface in self.interfaces:
			interface.writeBCs(TEMP)

	def readCouplingData(self):
		for interface in self.interfaces:
			interface.readAndUpdateBCs()

	def writeCheckpoint(self):
		if self.precice.requires_writing_checkpoint():
			# Do nothing
			pass

	def readCheckpoint(self):
		if self.precice.requires_reading_checkpoint():
			# Do nothing
			pass

	def isCouplingTimestepComplete(self):
		return self.precice.is_time_window_complete()

	def advance(self):
		self.precice.advance(self.preciceDt)

	def finalize(self):
		self.precice.finalize()

	def getMaxTimeStepSize(self):
		return self.precice.get_max_time_step_size()


class Interface:
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

		self.readData = dict()

		self.writeDataNames = []

		self.MESH = MESH
		# Shifted mesh (contains only the interface, and is shifted by delta in the direction opposite to the normal)
		self.SHMESH = SHMESH
		self.MODEL = MODEL
		self.MAT = MAT
		self.LOAD = None
		self.LOADS = []
		self.mesh = partition.MAIL_PY()
		self.mesh.FromAster(MESH)

		self.configure(names)

	def configure(self, names):

		self.groupName = names["patch"]

		# In Code_Aster, write-data is located at the nodes
		self.nodesMeshName = names["write-mesh"]
		# and read-data is located at the face centers
		self.faceCentersMeshName = names["read-mesh"]

		self.computeNormals()

		nodes_mesh_dims = self.precice.get_mesh_dimensions(self.nodesMeshName)
		face_centers_mesh_dims = self.precice.get_mesh_dimensions(self.faceCentersMeshName)

		if nodes_mesh_dims != face_centers_mesh_dims:
			raise ValueError("Dimensions of the read mesh and the write mesh do not match")
		else:
			# Set dimensions to nodes mesh dimensions
			dims = nodes_mesh_dims

		self.nodeCoordinates = np.array([p for p in self.SHMESH.sdj.COORDO.VALE.get()])
		self.nodeCoordinates = np.resize(self.nodeCoordinates, (int(len(self.nodeCoordinates)/dims), dims))
		self.shiftMesh()

		self.faces = [self.mesh.correspondance_mailles[idx] for idx in self.mesh.gma[self.groupName]]
		self.connectivity = [self.mesh.co[idx] for idx in self.mesh.gma[self.groupName]]
		self.faceCenterCoordinates = np.array([np.array([self.mesh.cn[node] for node in elem]).mean(0) for elem in self.connectivity])
		self.setVertices()

		self.setDataNames(names)

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

		dims = self.precice.get_mesh_dimensions(self.nodesMeshName)

		self.normals = np.resize(np.array(self.normals), (int(len(self.normals)/dims), dims))
		DETRUIRE(CONCEPT=({"NOM": N}, {"NOM": DUMMY}))

	def setVertices(self):
		# Nodes
		self.preciceNodeIndices = self.precice.set_mesh_vertices(self.nodesMeshName,  self.nodeCoordinates)
		# Face centers
		self.preciceFaceCenterIndices = self.precice.set_mesh_vertices(self.faceCentersMeshName, self.faceCenterCoordinates)

	def setDataNames(self, names):
		for writeDataName in names["write-data"]:
			if writeDataName.find("Heat-Transfer-Coefficient-Solid") >= 0:
				self.writeDataNames.append(writeDataName)
			elif writeDataName.find("Sink-Temperature-Solid") >= 0:
				self.writeDataNames.append(writeDataName)
		for readDataName in names["read-data"]:
			if readDataName.find("Heat-Transfer-Coefficient-Fluid") >= 0:
				self.readData[readDataName] = None
			elif readDataName.find("Sink-Temperature-Fluid") >= 0:
				self.readData[readDataName] = None

	def getPreciceNodeIndices(self):
		return self.preciceNodeIndices

	def getPreciceFaceCenterIndices(self):
		return self.preciceFaceCenterIndices

	def getNodeCoordinates(self):
		return self.nodeCoordinates

	def getFaceCenterCoordinates(self):
		return self.faceCenterCoordinates

	def getNormals(self):
		return self.normals

	def createBCs(self):
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

	def updateBCs(self, temp, hCoeff):
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
		dt = self.precice.get_max_time_step_size()
		for name in self.readData.keys():
			self.readData[name] = self.precice.read_data(self.faceCentersMeshName, name, self.preciceFaceCenterIndices, dt)

		self.updateBCs(self.readData["Sink-Temperature-Fluid"], self.readData["Heat-Transfer-Coefficient-Fluid"])

	def writeBCs(self, TEMP):
		writeTemp, writeHCoeff = self.getBoundaryValues(TEMP)

		self.precice.write_data(self.nodesMeshName, self.writeDataNames[0], self.preciceNodeIndices, writeHCoeff)
		self.precice.write_data(self.nodesMeshName, self.writeDataNames[1],  self.preciceNodeIndices, writeTemp)

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
		dims = self.precice.get_mesh_dimensions(self.nodesMeshName)
		coords = [p for p in self.SHMESH.sdj.COORDO.VALE.get()]
		for i in range(len(self.normals)):
			for c in range(dims):
				coords[i*dims + c] = coords[i*dims + c] - self.normals[i][c] * self.delta
		self.SHMESH.sdj.COORDO.VALE.changeJeveuxValues(len(coords), tuple(range(1, len(coords)+1)), tuple(coords), tuple(coords), 1)

	def updateConductivity(self, T):
		if self.isNonLinear:
			self.conductivity = [self.MAT.RCVALE("THER_NL", nompar="TEMP", valpar=t, nomres="LAMBDA")[0][0] for t in T]
			self.isConductivityInitialized = True
		elif not self.isConductivityInitialized:
			self.conductivity = [self.MAT.RCVALE("THER", nompar="TEMP", valpar=t, nomres="LAMBDA")[0][0] for t in T]
			self.isConductivityInitialized = True
		# Note: RCVALE returns ((LAMBDA,),(0,)), therefore we use [0][0] to extract the value of LAMBDA
