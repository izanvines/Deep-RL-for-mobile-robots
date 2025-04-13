import numpy as np

from gym import utils
from gym.envs.mujoco import MuJocoPyEnv
from gym.spaces import Box


class MuSHREnv(MuJocoPyEnv, utils.EzPickle):
    """
    ### Description
    "Rover" es un vehiculo de 4 ruedas. El objetivo es mover el robot hasta que se acerque al objetivo que aparecerá 
    de manera aleeatoria en una posicion

    ### Action Space
    `Box([-0.38 -0.3 ], [0.38 0.3 ], (2,), float32)`. Ana accion es tanto el giro como la velocidad del robot.

    | Num | Action                                                   | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit |
    |-----|----------------------------------------------------------|-------------|-------------|--------------------------|-------|------|
    | 0   | Giro de las ruedas delanteras                            | -0.38 | 0.38 |   | pos | Pos o ang (ni idea) |
    | 1   | Velocidad aplicada a las 4 ruedas (avance o retroceso)   | -0.3  | 0.3  |   | vel | (ni idea Km/h o m/s) |
    
    ### Observation Space
    
    ### Rewards

    ### Starting State

    ### Episode End

    ### Version History
    
    """
        
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20, #No entiendo porque solo funciona con 20
    }

    #Inicializacion del entorno
    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        self.last_action = np.zeros(2, dtype=np.float64)  # Inicializa acción previa
        observation_space = Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64)
        MuJocoPyEnv.__init__(
            self, "one_car.xml", 5, observation_space=observation_space, **kwargs
        )     

    def step(self, a):
        vec = self.get_body_com("buddy") - self.get_body_com("target")
        distance_to_target = np.linalg.norm(vec)

        #--------------------------------- CALCULO DE LA RECOMPENSA --------------------------------- 
        # Obtener distancia previa o inicializarla en infinito
        previous_distance = getattr(self, "previous_distance", np.inf)
        
        #Calcular progreso hacia el objetivo
        if np.isinf(previous_distance):  # Primer paso
            progress = 0.0  # No hay progreso aún
        else:
            progress = previous_distance - distance_to_target

        # Actualizar distancia previa
        self.previous_distance = distance_to_target

        #Recompensa basada en distancia (más cerca => más recompensa)
        weight_dist = 1.0
        reward_dist = weight_dist * np.exp(-distance_to_target)

        # Penalización suave por acciones grandes
        reward_ctrl = -np.square(a).sum() * 5 #Cuanto mas grande el valor más aumenta y mas penaliza...
        # Incentivar el progreso hacia el objetivo
        reward_progress = 500.0 * progress

        # Penalización por tiempo (pequeña cantidad negativa por cada paso)
        reward_time_penalty = -0.1  

        reward = reward_dist + reward_ctrl  + reward_progress + reward_time_penalty


        #Parámetro de distancia mínima para alcanzar el objetivo
        min_distance = 0.3
        # Recompensa adicional si alcanza el objetivo
        terminated = False
        if distance_to_target < min_distance:
            reward += 500.0  # Recompensa extra por alcanzar el objetivo
            terminated = True  # Finaliza el episodio

        #---------------------------------------------------------------------------------------------

        self.do_simulation(a, self.frame_skip) #Aplicar la accion "a" sobre la simulacion

        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()

        self.last_action = a  # Guarda acción actual como la "última"

        return (
            ob,
            reward,
            terminated,
            False,
            dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl),
        )

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 0
        
    def reset_model(self): 
       
        qpos = self.init_qpos

        # Generar una posición aleatoria dentro del rango de distancia especificado
        min_distance = 1.5
        max_distance = 4.0

        while True:
            # Generar una posición aleatoria dentro del cuadrado que engloba el círculo
            random_pos = self.np_random.uniform(low=-max_distance, high=max_distance, size=2)
            distance = np.linalg.norm(random_pos)
            
            # Aceptar solo posiciones dentro del anillo (entre los radios mínimo y máximo)
            if min_distance <= distance <= max_distance:
                break

        self.goal = random_pos

        qpos[-2:] = self.goal
               
        #Todas las velocidades a 0, partimos del reposo y el objetivo no se mueve
        qvel = self.init_qvel 

        self.set_state(qpos, qvel)

        self.last_action = np.zeros(2, dtype=np.float64)

        return self._get_obs()

    def _get_obs(self):
        # Posición relativa entre el rover y el objetivo (3 xyz)
        rel_pos = self.get_body_com("buddy") - self.get_body_com("target")  
        
        # Distancia al objetivo (1 dist)
        dist_to_target = [np.linalg.norm(self.get_body_com("buddy") - self.get_body_com("target"))]
        
        # Posición absoluta del rover (3 xyz)
        pos_buddy = self.get_body_com("buddy")  
        
        # Orientación del rover (1 angle)
        orientation_buddy = [np.arctan2(self.get_body_com("buddy")[1], self.get_body_com("buddy")[0])]
        
        # Posición del objetivo (3 xyz)
        pos_target = self.get_body_com("target")
        
        return np.concatenate(
            [
                rel_pos,           # Vector al objetivo (3)
                dist_to_target,    # Distancia al objetivo (1)
                pos_buddy,         # Posición del rover (3)
                orientation_buddy, # Orientación del rover (1)
                pos_target,        # Posición del objetivo (3)
                self.last_action   # Acción previa (2)
            ]
        )

