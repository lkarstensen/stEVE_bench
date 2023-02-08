import logging
import Sofa
import SofaRuntime  # pylint: disable=unused-import
import numpy as np
import pygame
import gymnasium as gym
import importlib


from .aorticarch import AorticArch
from .jwire import JWire


class Simulation:
    def __init__(
        self,
        vessel_tree: AorticArch,
        stop_device_at_tree_end: bool = True,
        image_frequency: float = 7.5,
        dt_simulation: float = 0.006,
        normalize_action: bool = False,
        target_size: bool = 5,
        init_visual: bool = False,
    ) -> None:
        self.logger = logging.getLogger(self.__module__)

        self.vessel_tree = vessel_tree
        self.image_frequency = image_frequency
        self.dt_simulation = dt_simulation
        self.stop_device_at_tree_end = stop_device_at_tree_end
        self.init_visual = init_visual
        self.normalize_action = normalize_action
        self.target_size = target_size
        grey_value = 0.3
        self.device = JWire(color=[grey_value, grey_value, grey_value])

        self.simulation_error = False
        self.velocity_limit = np.array(self.device.velocity_limit, dtype=np.float32)
        self.last_action = np.zeros_like(self.velocity_limit, dtype=np.float32)
        self._sofa_initialized = False
        self.initialized_last_reset = True
        self.root = None
        self._beam_mechanics = None
        self.initial_orientation = None
        self._camera = None
        self.vessel_object = None
        if init_visual:
            self.display_size = (1000, 1000)
            self._target = None
            self._sofa_gl = None
            self._opengl_gl = None
            self._opengl_glu = None

    @property
    def tracking_space(self) -> gym.spaces.Box:
        return self.vessel_tree.coordinate_space

    @property
    def action_space(self) -> gym.spaces.Box:
        low = np.array([-1, -1]) if self.normalize_action else -self.velocity_limit
        high = np.array([1, 1]) if self.normalize_action else self.velocity_limit
        return gym.spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32))

    @property
    def tracking(self) -> np.ndarray:
        tracking = self._beam_mechanics.DOFs.position.value[:, 0:3][::-1]
        if np.any(np.isnan(tracking[0])):
            self.logger.warning("Tracking is NAN, resetting devices")
            self.simulation_error = True
            self._reset_sofa_devices()
            tracking = self._beam_mechanics.DOFs.position.value[:, 0:3][::-1]
        return tracking

    @property
    def device_length_inserted(self) -> float:
        return self._beam_mechanics.DeployController.xtip.value[0]

    @property
    def device_length_maximum(self) -> float:
        return self.device.length

    @property
    def device_rotation(self) -> float:
        try:
            rot = self._beam_mechanics.DeployController.rotationInstrument.value[0]
        except AttributeError:
            rot = 0.0
        return rot

    @property
    def device_diameter(self) -> float:
        return self.device.radius * 2

    def step(self, action: np.ndarray) -> None:
        action = np.array(action).reshape(self.action_space.shape)
        tip = self.tracking[0]
        if (
            self.stop_device_at_tree_end
            and self.vessel_tree.at_tree_end(tip)
            and action[0] > 0
            and self.device_length_inserted > 10
        ):
            action[0] = 0.0
        velocity_limit = 1 if self.normalize_action else self.velocity_limit
        action = np.clip(action, -velocity_limit, velocity_limit)
        self.last_action = action
        if self.normalize_action:
            action = action * self.velocity_limit
        for _ in range(int((1 / self.image_frequency) / self.dt_simulation)):
            self._do_sofa_step(action)

    def reset(self, target: np.ndarray) -> None:

        if not self._sofa_initialized:
            self._init_sofa()
            self._sofa_initialized = True
        self._reset_sofa_devices()
        self.simulation_error = False
        if self.init_visual:
            self._target.mech_obj.translation = [
                target[0],
                target[1],
                target[2],
            ]
            Sofa.Simulation.init(self._target)

    def render(self) -> None:
        if self.init_visual:
            return self._render()

    def close(self):
        self._unload_sofa()
        if self.init_visual:
            pygame.quit()  # pylint: disable=no-member

    def _unload_sofa(self):
        Sofa.Simulation.unload(self.root)
        self._sofa_initialized = False

    def _do_sofa_step(self, action):
        trans = action[0]
        rot = action[1]
        tip = self._beam_mechanics.DeployController.xtip
        tip[0] += float(trans * self.root.dt.value)
        self._beam_mechanics.DeployController.xtip = tip
        tip_rot = self._beam_mechanics.DeployController.rotationInstrument
        tip_rot[0] += float(rot * self.root.dt.value)
        self._beam_mechanics.DeployController.rotationInstrument = tip_rot
        Sofa.Simulation.animate(self.root, self.root.dt.value)

    def _reset_sofa_devices(self):

        xtip = self._beam_mechanics.DeployController.xtip.value
        self._beam_mechanics.DeployController.xtip.value = xtip * 0.0
        rot_instr = self._beam_mechanics.DeployController.rotationInstrument.value
        self._beam_mechanics.DeployController.rotationInstrument.value = rot_instr * 0.0
        self._beam_mechanics.DeployController.indexFirstNode.value = 0
        Sofa.Simulation.reset(self.root)

    def _init_sofa(self):

        self.root = Sofa.Core.Node()
        self._load_plugins()
        self.root.gravity = [0.0, 0.0, 0.0]
        self.root.dt = self.dt_simulation
        self._basic_setup()
        self._add_vessel_tree()
        self._add_device()
        if self.init_visual:
            self._add_visual_nodes_and_camera()
        Sofa.Simulation.init(self.root)
        if self.init_visual:
            self._sofa_gl = importlib.import_module("Sofa.SofaGL")
            self._opengl_gl = importlib.import_module("OpenGL.GL")
            self._opengl_glu = importlib.import_module("OpenGL.GLU")

            self._init_display()
        self.logger.info("Sofa Initialized")

    def _load_plugins(self):
        self.root.addObject(
            "RequiredPlugin",
            pluginName="\
            BeamAdapter\
            Sofa.Component.AnimationLoop\
            Sofa.Component.Collision.Detection.Algorithm\
            Sofa.Component.Collision.Detection.Intersection\
            Sofa.Component.LinearSolver.Direct\
            Sofa.Component.IO.Mesh",
        )

    def _basic_setup(self):
        self.root.addObject("FreeMotionAnimationLoop")
        self.root.addObject("DefaultPipeline", draw="0", depth="6", verbose="1")
        self.root.addObject("BruteForceBroadPhase")
        self.root.addObject("BVHNarrowPhase")
        self.root.addObject(
            "LocalMinDistance",
            contactDistance=0.3,
            alarmDistance=0.5,
            angleCone=0.02,
            name="localmindistance",
        )
        self.root.addObject(
            "DefaultContactManager", response="FrictionContactConstraint"
        )
        # self.root.addObject("DefaultCollisionGroupManager", name="Group")
        self.root.addObject(
            "LCPConstraintSolver",
            mu=0.1,
            tolerance=1e-4,
            maxIt=2000,
            build_lcp=False,
        )

    def _add_vessel_tree(self):
        vessel_object = self.root.addChild("vesselTree")
        vessel_object.addObject(
            "MeshObjLoader",
            filename=self.vessel_tree.mesh_path,
            flipNormals=False,
            name="meshLoader",
        )
        vessel_object.addObject(
            "MeshTopology",
            position="@meshLoader.position",
            triangles="@meshLoader.triangles",
        )
        vessel_object.addObject("MechanicalObject", name="dofs", src="@meshLoader")
        vessel_object.addObject("TriangleCollisionModel", moving=False, simulated=False)
        vessel_object.addObject("LineCollisionModel", moving=False, simulated=False)
        self.vessel_object = vessel_object

    def _add_device(self):
        topo_lines = self.root.addChild("EdgeTopology")
        rest_shape_name = self.device.name + "_rest_shape"
        if not self.device.is_a_procedural_shape:
            topo_lines.addObject(
                "MeshObjLoader",
                filename=self.device.mesh_path,
                name="loader",
            )
        topo_lines.addObject(
            "WireRestShape",
            name=rest_shape_name,
            straightLength=self.device.straight_length,
            length=self.device.length,
            spireDiameter=self.device.spire_diameter,
            radiusExtremity=self.device.radius_extremity,
            youngModulusExtremity=self.device.young_modulus_extremity,
            massDensityExtremity=self.device.mass_density_extremity,
            radius=self.device.radius,
            youngModulus=self.device.young_modulus,
            massDensity=self.device.mass_density,
            poissonRatio=self.device.poisson_ratio,
            keyPoints=self.device.key_points,
            densityOfBeams=self.device.density_of_beams,
            numEdgesCollis=self.device.num_edges_collis,
            numEdges=self.device.num_edges,
            spireHeight=self.device.spire_height,
            printLog=True,
            template="Rigid3d",
        )
        topo_lines.addObject("EdgeSetTopologyContainer", name="meshLines")
        topo_lines.addObject("EdgeSetTopologyModifier", name="Modifier")
        topo_lines.addObject(
            "EdgeSetGeometryAlgorithms", name="GeomAlgo", template="Rigid3d"
        )
        topo_lines.addObject("MechanicalObject", name="dofTopo2", template="Rigid3d")

        beam_mechanics = self.root.addChild("BeamModel")
        beam_mechanics.addObject(
            "EulerImplicitSolver", rayleighStiffness=0.2, rayleighMass=0.1
        )
        beam_mechanics.addObject(
            "BTDLinearSolver", verification=False, subpartSolve=False, verbose=False
        )
        nx = sum(self.device.density_of_beams) + 1
        beam_mechanics.addObject(
            "RegularGridTopology",
            name="MeshLines",
            nx=nx,
            ny=1,
            nz=1,
            xmax=0.0,
            xmin=0.0,
            ymin=0,
            ymax=0,
            zmax=0,
            zmin=0,
            p0=[0, 0, 0],
        )
        beam_mechanics.addObject(
            "MechanicalObject",
            showIndices=False,
            name="DOFs",
            template="Rigid3d",
        )
        beam_mechanics.addObject(
            "WireBeamInterpolation",
            name="BeamInterpolation",
            WireRestShape="@../EdgeTopology/" + rest_shape_name,
            radius=self.device.radius,
            printLog=False,
        )
        beam_mechanics.addObject(
            "AdaptiveBeamForceFieldAndMass",
            name="BeamForceField",
            massDensity=0.00000155,
            interpolation="@BeamInterpolation",
        )

        insertion_pose = self._calculate_insertion_pose(
            self.vessel_tree.insertion.position,
            self.vessel_tree.insertion.direction,
        )

        beam_mechanics.addObject(
            "InterventionalRadiologyController",
            name="DeployController",
            template="Rigid3d",
            instruments="BeamInterpolation",
            startingPos=insertion_pose,
            xtip=[0],
            printLog=True,
            rotationInstrument=[0],
            speed=0.0,
            listening=True,
            controlledInstrument=0,
        )
        beam_mechanics.addObject(
            "LinearSolverConstraintCorrection", wire_optimization="true", printLog=False
        )
        beam_mechanics.addObject("FixedConstraint", indices=0, name="FixedConstraint")
        beam_mechanics.addObject(
            "RestShapeSpringsForceField",
            points="@DeployController.indexFirstNode",
            angularStiffness=1e8,
            stiffness=1e8,
            external_points=0,
            external_rest_shape="@DOFs",
        )
        self._beam_mechanics = beam_mechanics

        beam_collis = beam_mechanics.addChild("CollisionModel")
        beam_collis.activated = True
        beam_collis.addObject("EdgeSetTopologyContainer", name="collisEdgeSet")
        beam_collis.addObject("EdgeSetTopologyModifier", name="colliseEdgeModifier")
        beam_collis.addObject("MechanicalObject", name="CollisionDOFs")
        beam_collis.addObject(
            "MultiAdaptiveBeamMapping",
            controller="../DeployController",
            useCurvAbs=True,
            printLog=False,
            name="collisMap",
        )
        beam_collis.addObject("LineCollisionModel", proximity=0.0)
        beam_collis.addObject("PointCollisionModel", proximity=0.0)

    def _render(self):
        Sofa.Simulation.updateVisual(self.root)
        gl = self._opengl_gl
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        self._opengl_glu.gluPerspective(
            self._camera.fieldOfView.value,
            (self._camera.widthViewport.value / self._camera.heightViewport.value),
            self._camera.zNear.value,
            self._camera.zFar.value,
        )
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        camera_mvm = self._camera.getOpenGLModelViewMatrix()
        gl.glMultMatrixd(camera_mvm)
        self._sofa_gl.draw(self.root)
        gl = self._opengl_gl
        height = self._camera.heightViewport.value
        width = self._camera.widthViewport.value

        buffer = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        image_array = np.fromstring(buffer, np.uint8)

        if image_array != []:
            image = image_array.reshape(height, width, 3)
            image = np.flipud(image)[:, :, :3]
        else:
            image = np.zeros((height, width, 3))
        pygame.display.flip()
        return np.copy(image)

    def _add_visual_nodes_and_camera(self):

        self.root.addObject(
            "RequiredPlugin",
            pluginName="\
            Sofa.GL.Component.Rendering3D\
            Sofa.GL.Component.Shader",
        )

        # VesselTree
        visu = self.vessel_object.addChild("Visu")
        visu.addObject(
            "OglModel",
            name="VisualModel",
            src="@../meshLoader",
            material="texture Ambient 1 0.2 0.2 0.2 0.0 Diffuse 1 1.0 1.0 1.0 1.0 Specular 1 1.0 1.0 1.0 1.0 Emissive 0 0.15 0.05 0.05 0.0 Shininess 1 10",
            color=[0.7, 0.0, 0.0, 0.3],
        )
        visu.addObject(
            "BarycentricMapping",
            name="VisualMapping",
            input="@../dofs",
            output="@VisualModel",
        )

        # Guidewire
        visu_node = self.root.addChild("Visu")
        visu_node.addObject("MechanicalObject", name="Quads")
        visu_node.addObject("QuadSetTopologyContainer", name="ContainerTube")
        visu_node.addObject("QuadSetTopologyModifier", name="Modifier")
        visu_node.addObject(
            "QuadSetGeometryAlgorithms",
            name="GeomAlgo",
            template="Vec3d",
        )
        visu_node.addObject(
            "Edge2QuadTopologicalMapping",
            nbPointsOnEachCircle="10",
            radius=self.device.radius,
            flipNormals="true",
            input="@../EdgeTopology/meshLines",
            output="@ContainerTube",
        )
        visu_node.addObject(
            "AdaptiveBeamMapping",
            interpolation="@../BeamModel/BeamInterpolation",
            name="VisuMap",
            output="@Quads",
            isMechanical="false",
            input="@../BeamModel/DOFs",
            useCurvAbs="1",
            printLog="0",
        )
        visu_ogl = visu_node.addChild("VisuOgl")
        visu_ogl.addObject(
            "OglModel",
            color=self.device.color,
            quads="@../ContainerTube.quads",
            src="@../ContainerTube",
            material="texture Ambient 1 0.2 0.2 0.2 0.0 Diffuse 1 1.0 1.0 1.0 1.0 Specular 1 1.0 1.0 1.0 1.0 Emissive 0 0.15 0.05 0.05 0.0 Shininess 1 20",
            name="Visual",
        )
        visu_ogl.addObject(
            "IdentityMapping",
            input="@../Quads",
            output="@Visual",
        )

        # Target
        target_node = self.root.addChild("main_target")
        target_node.addObject(
            "MeshSTLLoader",
            name="loader",
            triangulate=True,
            filename="/Users/lennartkarstensen/stacie/eve/eve/visualisation/meshes/unit_sphere.stl",
            scale=self.target_size,
            translation=[0, 0, 0],
        )
        (
            target_node.addObject(
                "MechanicalObject",
                src="@loader",
                translation=(0, 0, 0),
                template="Rigid3d",
                name="mech_obj",
            )
        )
        target_node.addObject(
            "OglModel",
            src="@loader",
            color=[0.0, 0.9, 0.5, 0.4],
            translation=[0.0, 0.0, 0.0 - self.target_size],
            material="texture Ambient 1 0.2 0.2 0.2 0.0 Diffuse 1 1.0 1.0 1.0 1.0 Specular 1 1.0 1.0 1.0 1.0 Emissive 0 0.15 0.05 0.05 0.0 Shininess 1 20",
            name="ogl_model",
        )
        target_node.addObject("RigidMapping", input="@mech_obj")
        self._target = target_node

        # Camera
        self.root.addObject("DefaultVisualManagerLoop")
        self.root.addObject(
            "VisualStyle",
            displayFlags="showVisualModels\
                hideBehaviorModels\
                hideCollisionModels\
                hideWireframe\
                hideMappings\
                hideForceFields",
        )
        self.root.addObject("LightManager")
        self.root.addObject("DirectionalLight", direction=[0, -1, 0])
        self.root.addObject("DirectionalLight", direction=[0, 1, 0])

        # TODO: Find out how to manipulate background. BackgroundSetting doesn't seem to work
        # self.root.addObject("BackgroundSetting", color=(0.5, 0.5, 0.5, 1.0))

        look_at = (
            self.vessel_tree.coordinate_space.high
            + self.vessel_tree.coordinate_space.low
        ) * 0.5
        distance_coefficient = 1.5
        distance = (
            np.linalg.norm(look_at - self.vessel_tree.coordinate_space.low)
            * distance_coefficient
        )
        position = look_at + np.array([0.0, -distance, 0.0])
        scene_radius = np.linalg.norm(
            self.vessel_tree.coordinate_space.high
            - self.vessel_tree.coordinate_space.low
        )
        dist_cam_to_center = np.linalg.norm(position - look_at)
        z_clipping_coeff = 5
        z_near_coeff = 0.01
        z_near = dist_cam_to_center - scene_radius
        z_far = (z_near + 2 * scene_radius) * 2
        z_near = z_near * z_near_coeff
        z_min = z_near_coeff * z_clipping_coeff * scene_radius
        if z_near < z_min:
            z_near = z_min
        field_of_view = 70
        look_at = np.array(look_at)
        position = np.array(position)

        self._camera = self.root.addObject(
            "Camera",
            name="camera",
            lookAt=look_at,
            position=position,
            fieldOfView=field_of_view,
            widthViewport=self.display_size[0],
            heightViewport=self.display_size[1],
            zNear=z_near,
            zFar=z_far,
            fixedLookAt=False,
        )

    def _init_display(self):
        # pylint: disable=no-member

        pygame.display.init()
        flags = pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE
        pygame.display.set_mode(self.display_size, flags)

        gl = self._opengl_gl
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)
        Sofa.SofaGL.glewInit()
        Sofa.Simulation.initVisual(self.root)
        Sofa.Simulation.initTextures(self.root)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        self._opengl_glu.gluPerspective(
            self._camera.fieldOfView.value,
            (self._camera.widthViewport.value / self._camera.heightViewport.value),
            self._camera.zNear.value,
            self._camera.zFar.value,
        )
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    @staticmethod
    def _calculate_insertion_pose(
        insertion_point: np.ndarray, insertion_direction: np.ndarray
    ):

        insertion_direction = insertion_direction / np.linalg.norm(insertion_direction)
        original_direction = np.array([1.0, 0.0, 0.0])
        if np.all(insertion_direction == original_direction):
            w0 = 1.0
            xyz0 = [0.0, 0.0, 0.0]
        elif np.all(np.cross(insertion_direction, original_direction) == 0):
            w0 = 0.0
            xyz0 = [0.0, 1.0, 0.0]
        else:
            half = (original_direction + insertion_direction) / np.linalg.norm(
                original_direction + insertion_direction
            )
            w0 = np.dot(original_direction, half)
            xyz0 = np.cross(original_direction, half)
        xyz0 = list(xyz0)
        pose = list(insertion_point) + list(xyz0) + [w0]
        return pose
