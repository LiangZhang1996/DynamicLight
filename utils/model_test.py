from .config import DIC_AGENTS
from copy import deepcopy
from .cityflow_env import CityFlowEnv
import json
import os



def test(model_dir, cnt_round, run_cnt, _dic_traffic_env_conf):
    dic_traffic_env_conf = deepcopy(_dic_traffic_env_conf)
    records_dir = model_dir.replace("model", "records")
    model_round = "round_%d" % cnt_round
    dic_path = {"PATH_TO_MODEL": model_dir, "PATH_TO_WORK_DIRECTORY": records_dir}
    with open(os.path.join(records_dir, "agent.conf"), "r") as f:
        dic_agent_conf = json.load(f)
    if os.path.exists(os.path.join(records_dir, "anon_env.conf")):
        with open(os.path.join(records_dir, "anon_env.conf"), "r") as f:
            dic_traffic_env_conf = json.load(f)
    dic_traffic_env_conf["RUN_COUNTS"] = run_cnt

    dic_agent_conf["EPSILON"] = 0
    dic_agent_conf["MIN_EPSILON"] = 0

    agents = []
    for i in range(dic_traffic_env_conf['NUM_AGENTS']):
        agent_name = dic_traffic_env_conf["MODEL_NAME"]
        agent = DIC_AGENTS[agent_name](
            dic_agent_conf=dic_agent_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            cnt_round=0,
            intersection_id=str(i),
        )
        agents.append(agent)
    try:
        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            agents[i].load_network("{0}_inter_{1}".format(model_round, agents[i].intersection_id))
            agents[i].cnt_round = cnt_round
        path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", model_round)
        if not os.path.exists(path_to_log):
            os.makedirs(path_to_log)
        env = CityFlowEnv(
            path_to_log=path_to_log,
            path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=dic_traffic_env_conf
        )
        total_time = dic_traffic_env_conf["RUN_COUNTS2"]
        state, step_time, list_need = env.reset()
        while step_time < total_time:
            for i in range(dic_traffic_env_conf["NUM_AGENTS"]):
                phase_action, duration_action = agents[i].choose_action(state, list_need)
            next_state, step_time, list_need = env.step(phase_action, duration_action)
            state = next_state
        env.batch_log_2()
        env.end_cityflow()
    except:
        print("============== error occurs in model_test ============")
