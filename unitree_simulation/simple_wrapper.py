import numpy as np
from unitree_description.loader import loadGo2
from unitree_description import GO2_DESCRIPTION_SRDF_PATH
import hppfcl
import pinocchio as pin
import simple
from unitree_simulation.abstract_wrapper import AbstractSimulatorWrapper
import threading
import queue


class SimpleSimulator:
    def __init__(self, model, geom_model, q0, args):
        self.model = model
        self.geom_model = geom_model
        self.args = args

        self.data = self.model.createData()
        self.geom_data = self.geom_model.createData()

        for col_req in self.geom_data.collisionRequests:
            col_req: hppfcl.CollisionRequest
            col_req.security_margin = 0.0
            col_req.break_distance = 0.0
            col_req.gjk_tolerance = 1e-6
            col_req.epa_tolerance = 1e-6
            col_req.gjk_initial_guess = hppfcl.GJKInitialGuess.CachedGuess
            col_req.gjk_variant = hppfcl.GJKVariant.DefaultGJK

        for patch_req in self.geom_data.contactPatchRequests:
            patch_req.setPatchTolerance(args["patch_tolerance"])

        # Simulation parameters
        self.simulator = simple.Simulator(model, self.data, geom_model, self.geom_data)
        # admm
        self.simulator.admm_constraint_solver_settings.absolute_precision = args["tol"]
        self.simulator.admm_constraint_solver_settings.relative_precision = args["tol_rel"]
        self.simulator.admm_constraint_solver_settings.max_iter = args["maxit"]
        self.simulator.admm_constraint_solver_settings.mu = args["mu_prox"]
        # pgs
        self.simulator.pgs_constraint_solver_settings.absolute_precision = args["tol"]
        self.simulator.pgs_constraint_solver_settings.relative_precision = args["tol_rel"]
        self.simulator.pgs_constraint_solver_settings.max_iter = args["maxit"]
        #
        self.simulator.warm_start_constraint_forces = args["warm_start"]
        self.simulator.measure_timings = True
        # Contact patch settings
        self.simulator.constraint_problem.setMaxNumberOfContactsPerCollisionPair(args["max_patch_size"])
        # Baumgarte settings
        contact_constraints = self.simulator.constraint_problem.frictional_point_constraint_models
        for i in range(len(contact_constraints)):
            contact_constraints[i].baumgarte_corrector_parameters.Kp = args["Kp"]
            contact_constraints[i].baumgarte_corrector_parameters.Kd = args["Kd"]
        if args["admm_update_rule"] == "spectral":
            self.simulator.admm_constraint_solver_settings.admm_update_rule = pin.ADMMUpdateRule.SPECTRAL
        elif args["admm_update_rule"] == "linear":
            self.simulator.admm_constraint_solver_settings.admm_update_rule = pin.ADMMUpdateRule.LINEAR
        else:
            update_rule = args["admm_update_rule"]
            print(f"ERROR - no match for admm update rule {update_rule}")
            exit(1)
        self.dt = args["dt"]

        # Initialize robot state
        self.q = q0.copy()
        self.v = np.zeros(self.model.nv)
        self.a = np.zeros(self.model.nv)
        self.f_feet = np.zeros(4)
        self.foot_names = ["FR_foot_0", "FL_foot_0", "RR_foot_0", "RL_foot_0"]

        self.simulator.reset()

    def execute(self, tau):
        if self.args["contact_solver"] == "ADMM":
            self.simulator.step(self.q, self.v, tau, self.dt)
        else:
            self.simulator.stepPGS(self.q, self.v, tau, self.dt)

        self.q = self.simulator.qnew.copy()
        self.v = self.simulator.vnew.copy()
        self.a = self.simulator.anew.copy()

        # Detect contact through pair of collision
        self.f_feet = np.zeros(4)
        for cp_id in self.simulator.constraint_problem.pairs_in_collision:
            cp = self.simulator.geom_model.collisionPairs[cp_id]
            first = self.simulator.geom_model.geometryObjects[cp.first].name
            if first in self.foot_names:
                self.f_feet[self.foot_names.index(first)] = 39.4  # roughly 1/4 of the robot mass (0th order approx)
            second = self.simulator.geom_model.geometryObjects[cp.second].name
            if second in self.foot_names:
                self.f_feet[self.foot_names.index(second)] = 39.4  # roughly 1/4 of the robot mass (0th order approx)

        return self.q, self.v, self.a, self.f_feet


def setPhysicsProperties(geom_model: pin.GeometryModel, material: str, compliance: float):
    for gobj in geom_model.geometryObjects:
        if material == "ice":
            gobj.physicsMaterial.materialType = pin.PhysicsMaterialType.ICE
        elif material == "plastic":
            gobj.physicsMaterial.materialType = pin.PhysicsMaterialType.PLASTIC
        elif material == "wood":
            gobj.physicsMaterial.materialType = pin.PhysicsMaterialType.WOOD
        elif material == "metal":
            gobj.physicsMaterial.materialType = pin.PhysicsMaterialType.METAL
        elif material == "concrete":
            gobj.physicsMaterial.materialType = pin.PhysicsMaterialType.CONCRETE

        # Compliance
        gobj.physicsMaterial.compliance = compliance


def removeBVHModelsIfAny(geom_model: pin.GeometryModel):
    for gobj in geom_model.geometryObjects:
        gobj: pin.GeometryObject
        bvh_types = [hppfcl.BV_OBBRSS, hppfcl.BV_OBB, hppfcl.BV_AABB]
        ntype = gobj.geometry.getNodeType()
        if ntype in bvh_types:
            gobj.geometry.buildConvexHull(True, "Qt")
            gobj.geometry = gobj.geometry.convex


def addFloor(robot: pin.RobotWrapper):
    # Collision object
    floor_collision_shape = hppfcl.Halfspace(0, 0, 1, 0)
    M = pin.SE3.Identity()
    floor_collision_object = pin.GeometryObject("floor", 0, 0, M, floor_collision_shape)
    robot.collision_model.addGeometryObject(floor_collision_object)
    robot.collision_data = robot.collision_model.createData()

    # Recompute collision pairs
    robot.collision_model.removeAllCollisionPairs()
    robot.collision_model.addAllCollisionPairs()
    pin.removeCollisionPairs(robot.model, robot.collision_model, GO2_DESCRIPTION_SRDF_PATH)

    # Visual object
    floor_thickness = 0.01
    floor_visual_shape = hppfcl.Box(10, 10, floor_thickness)
    floor_pose = pin.XYZQUATToSE3([0, 0, -floor_thickness] + [0, 0, 0, 1])
    floor_visual_object = pin.GeometryObject("floor", 0, 0, floor_pose, floor_visual_shape)
    floor_visual_object.meshColor = 0.5 * np.ones(4)
    robot.visual_model.addGeometryObject(floor_visual_object)
    robot.visual_data = robot.visual_model.createData()


class SimpleWrapper(AbstractSimulatorWrapper):
    def __init__(self, node, timestep):
        ########################## Load robot model and geometry
        self.robot = loadGo2()
        self.rmodel = self.robot.model

        # Ignore friction and kinematics limits inside the simulator
        for i in range(self.rmodel.nq):
            self.rmodel.lowerPositionLimit[i] = np.finfo("d").min
            self.rmodel.upperPositionLimit[i] = np.finfo("d").max
        self.rmodel.lowerDryFrictionLimit[:] = 0
        self.rmodel.upperDryFrictionLimit[:] = 0

        # Load parameters from node
        self.params = {
            "viewer": node.declare_parameter("viewer", True).value,
            "Kp": node.declare_parameter("Kp", 0.0).value,
            "Kd": node.declare_parameter("Kd", 0.0).value,
            "compliance": node.declare_parameter("compliance", 0.0).value,
            "material": node.declare_parameter("material", "metal").value,
            "horizon": node.declare_parameter("horizon", 1000).value,
            "dt": timestep,
            "tol": node.declare_parameter("tol", 1e-6).value,
            "tol_rel": node.declare_parameter("tol_rel", 1e-6).value,
            "mu_prox": node.declare_parameter("mu_prox", 1e-4).value,
            "maxit": node.declare_parameter("maxit", 100).value,
            "warm_start": node.declare_parameter("warm_start", 1).value,
            "contact_solver": node.declare_parameter("contact_solver", "ADMM").value,
            "admm_update_rule": node.declare_parameter("admm_update_rule", "spectral").value,
            "max_patch_size": node.declare_parameter("max_patch_size", 2).value,
            "patch_tolerance": node.declare_parameter("patch_tolerance", 1e-2).value,
        }

        self.init_simple()

        # Prepare viewer
        if self.params["viewer"]:
            self.viewer_q_queue = queue.Queue(maxsize=1)  # Only keep the latest q
            self.viewer_stop_event = threading.Event()
            self.viewer_thread = threading.Thread(target=self._viewer_loop, daemon=True)
            self.viewer_thread.start()

    def _del__(self):
        self.viewer_stop_event.set()
        self.viewer_thread.join()

    def init_simple(self):
        # Start the robot in crouch pose 15cm above the ground
        initial_q = np.array([0, 0, 0.15, 0, 0, 0, 1, 0.0, 0.9, -2.5, 0.0, 0.9, -2.5, 0.0, 0.9, -2.5, 0, 0.9, -2.5])

        # Set simulation properties
        addFloor(self.robot)
        setPhysicsProperties(self.robot.collision_model, self.params["material"], self.params["compliance"])
        removeBVHModelsIfAny(self.robot.collision_model)

        # Unitree joint ordering (FR, FL, RR, RL)
        self.joint_order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

        # Create the simulator object
        self.simulator = SimpleSimulator(self.rmodel, self.robot.collision_model, initial_q, self.params)

    def _viewer_loop(self):
        self.robot.initViewer(open=True)
        self.robot.viz.loadViewerModel()
        self.robot.viz.display_collisions = True
        while not self.viewer_stop_event.is_set():
            try:
                q = self.viewer_q_queue.get(timeout=1.0)
                self.robot.display(q)
            except queue.Empty:
                continue

    def _viewer_async_display(self, q):
        if self.params["viewer"] and not self.viewer_q_queue.full():
            self.viewer_q_queue.put_nowait(q)

    def step(self, tau_cmd):
        # Change torque order from unitree to pinocchio
        torque_simu = np.zeros(self.rmodel.nv)
        for i in range(12):
            torque_simu[6 + i] = tau_cmd[self.joint_order[i]]

        # Execute step and get new state
        q_current, v_current, a_current, f_current = self.simulator.execute(torque_simu)

        # Display
        self._viewer_async_display(q_current)

        # Reorder state from pinocchio to unitree order
        q_unitree = q_current.copy()
        v_unitree = v_current.copy()
        a_unitree = a_current.copy()
        for i in range(12):
            q_unitree[7 + i] = q_current[7 + self.joint_order[i]]
            v_unitree[6 + i] = v_current[6 + self.joint_order[i]]
            a_unitree[6 + i] = a_current[6 + self.joint_order[i]]

        # Reorder contacts from (FL, FR, RR, RL) to (FR, FL, RR, RL)
        f_unitree = np.array([f_current[1], f_current[0], f_current[3], f_current[2]])

        return q_unitree, v_unitree, a_unitree, f_unitree
