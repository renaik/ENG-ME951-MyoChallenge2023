""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
import numpy as np
import gym

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import mat2euler, euler2quat, quat2euler


class RelocateEnvV0(BaseV0):
    DEFAULT_OBS_KEYS = ['hand_qpos', 'hand_qvel', 'obj_pos', 'goal_pos', 'pos_err', 'obj_rot', 'goal_rot', 'rot_err']
    RWD_KEYS_AND_WEIGHTS = {
        #"pos_dist": 100.0,
        #"goal_obj_rot": 1.0,
        "done": 0.0,
        "act_reg": 0.0,
        "sparse": 0.0,
        "solved": 0.0
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)


    def _setup(self,
            target_xyz_range,     # target position range (relative to initial pos)
            target_rxryrz_range,  # target rotation range (relative to initial rot)
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:list = RWD_KEYS_AND_WEIGHTS,
            pos_th = .025,        # position error threshold
            rot_th = 0.262,       # rotation error threshold
            drop_th = 0.50,       # drop height threshold
            lift_th = 0.02,       # lift heigh threshold
            contact_th = 0.005,   # contact object threshold
            reach_z_offset = 0.,  # reach z offset
            pos_z_offset = 0,     # pos z offset
            **kwargs,
        ):
        self.palm_sid = self.sim.model.site_name2id("S_grasp")
        self.tip0 = self.sim.model.site_name2id("THtip")
        self.tip1 = self.sim.model.site_name2id("IFtip")
        self.tip2 = self.sim.model.site_name2id("MFtip")
        self.tip3 = self.sim.model.site_name2id("RFtip")
        self.tip4 = self.sim.model.site_name2id("LFtip")
        self.object_sid = self.sim.model.site_name2id("object_o")
        self.goal_sid = self.sim.model.site_name2id("target_o")
        self.success_indicator_sid = self.sim.model.site_name2id("target_ball")
        self.goal_bid = self.sim.model.body_name2id("target")
        self.target_xyz_range = target_xyz_range
        self.target_rxryrz_range = target_rxryrz_range
        self.pos_th = pos_th
        self.rot_th = rot_th
        self.drop_th = drop_th
        self.lift_th = lift_th
        self.contact_th = contact_th
        self.reach_z_offset = reach_z_offset

        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       target_rxryrz_range=target_rxryrz_range,
                       pos_th=pos_th,
                       rot_th=rot_th,
                       drop_th=drop_th,
                       **kwargs,
        )


    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['tip0'] = sim.data.site_xpos[self.tip0]
        obs_dict['tip1'] = sim.data.site_xpos[self.tip1]
        obs_dict['tip2'] = sim.data.site_xpos[self.tip2]
        obs_dict['tip3'] = sim.data.site_xpos[self.tip3]
        obs_dict['tip4'] = sim.data.site_xpos[self.tip4]
        obs_dict['hand_qpos'] = sim.data.qpos[:-7].copy()
        obs_dict['hand_qvel'] = sim.data.qvel[:-6].copy()*self.dt
        obs_dict['obj_pos'] = sim.data.site_xpos[self.object_sid]
        obs_dict['goal_pos'] = sim.data.site_xpos[self.goal_sid]
        obs_dict['palm_pos'] = sim.data.site_xpos[self.palm_sid]
        obs_dict['pos_err'] = obs_dict['goal_pos'] - obs_dict['obj_pos']
        obs_dict['reach_err'] = obs_dict['palm_pos'] - obs_dict['obj_pos']
        obs_dict['obj_rot'] = mat2euler(np.reshape(sim.data.site_xmat[self.object_sid],(3,3)))
        obs_dict['goal_rot'] = mat2euler(np.reshape(sim.data.site_xmat[self.goal_sid],(3,3)))
        obs_dict['palm_rot'] = mat2euler(np.reshape(sim.data.site_xmat[self.palm_sid], (3,3)))
        obs_dict['rot_err'] = obs_dict['goal_rot'] - obs_dict['obj_rot']

        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        return obs_dict


    def get_reward_dict(self, obs_dict):
        # Sub-task 1: Reach.
        reach_dist = np.abs(np.linalg.norm(obs_dict['reach_err'] + np.array([0., 0., self.reach_z_offset]), axis=-1))
        reach_dist_xy = np.abs(np.linalg.norm(obs_dict['reach_err'][:,:,:2], axis=-1)) 
        reach_dist_z = np.abs(np.linalg.norm(obs_dict['reach_err'][:,:,2] + self.reach_z_offset, axis=-1))
        contact = np.abs(np.linalg.norm(obs_dict['reach_err'], axis=-1)) < self.contact_th

        # Sub-task 2: Grasp.
        close_hand_dist = 0
        grasp_obj_dist = 0
        for i in range(5):
           close_hand_dist += np.abs(np.linalg.norm(obs_dict['tip'+str(i)] - obs_dict['palm_pos'], axis=-1))
           grasp_obj_dist += np.abs(np.linalg.norm(obs_dict['tip'+str(i)] - obs_dict['obj_pos'], axis=-1))
        palm_obj_rot = np.abs(np.linalg.norm(obs_dict['palm_rot'] - obs_dict['obj_rot'], axis=-1))

        # Sub-task 3: Lift.
        lift = self.obs_dict['obj_pos'][:, :, 2] > self.lift_th

        # Sub-task 4: Relocate.
        pos_dist = np.abs(np.linalg.norm(obs_dict['pos_err'], axis=-1))

        # Sub-task 5: Drop.
        goal_obj_rot = np.abs(np.linalg.norm(obs_dict['rot_err'], axis=-1))
        act_mag = np.linalg.norm(obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        drop = reach_dist > self.drop_th

        epsilon = 1e-4
        rwd_dict = collections.OrderedDict((
            # Optional Keys.
            # Sub-task 1: Reach.
            ('reach_dist', -1.*reach_dist),
            ('reach_obj', -1.*(reach_dist + np.log(reach_dist + epsilon**2))),
            ('reach_obj_xy', -1.*(reach_dist_xy + np.log(reach_dist + epsilon**2))),
            ('reach_obj_z', -1.*(reach_dist_z + np.log(reach_dist + epsilon**2))),
            ('contact_obj', contact),
            
            # Sub-task 2: Grasp.
            ('close_hand_dist', 1.*close_hand_dist),
            ('grasp_obj_dist', -1.*grasp_obj_dist),
            ('palm_obj_rot', -1.*palm_obj_rot),
            ('close_hand', -1.*np.exp(-5.*close_hand_dist)),
            ('grasp_obj', 1.*np.exp(-1.*grasp_obj_dist)),

            # Sub-task 3: Lift.
            ('lift', lift),

            # Sub-task 4: Relocate.
            ('relocate_obj', np.exp(-5 * pos_dist)),

            # Sub-task 5: Drop.
            ('goal_obj_rot', -1.*goal_obj_rot),

            # Must keys.
            ('act_reg', -1.*act_mag),
            ('sparse', -goal_obj_rot - 10.0*pos_dist),
            ('solved', (pos_dist<self.pos_th) and (not drop)),
            ('done', drop),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)

        # Success Indicator.
        self.sim.model.site_rgba[self.success_indicator_sid, :2] = np.array([0, 2]) if rwd_dict['solved'] else np.array([2, 0])
        return rwd_dict


    def get_metrics(self, paths, successful_steps=5):
        """
        Evaluate paths and report metrics.
        """
        num_success = 0
        num_paths = len(paths)

        # Average sucess over entire env horizon.
        for path in paths:
            # Record success if solved for provided successful_steps.
            if np.sum(path['env_infos']['rwd_dict']['solved'] * 1.0) > successful_steps:
                num_success += 1
        score = num_success/num_paths

        # Average activations over entire trajectory (can be shorter than horizon, if done) realized.
        effort = -1.0*np.mean([np.mean(p['env_infos']['rwd_dict']['act_reg']) for p in paths])

        metrics = {
            'score': score,
            'effort':effort,
            }
        return metrics

    def reset(self, reset_qpos=None, reset_qvel=None):
        self.sim.model.body_pos[self.goal_bid] = self.np_random.uniform(**self.target_xyz_range)
        self.sim.model.body_quat[self.goal_bid] =  euler2quat(self.np_random.uniform(**self.target_rxryrz_range))
        obs = super().reset(reset_qpos, reset_qvel)
        return obs