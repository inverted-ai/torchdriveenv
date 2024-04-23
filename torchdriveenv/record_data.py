import logging

from torchdrivesim.simulator import RecordingWrapper, SimulatorInterface
from torchdrivesim.behavior.iai import IAIWrapper

logger = logging.getLogger(__name__)


class OfflineDataRecordingWrapper(RecordingWrapper):
    """
    Record agent_states and traffic_light_state_history.
    """

    def __init__(self, simulator: SimulatorInterface):

        def record_agent_states(simulator):
            iai_simulator = simulator.inner_simulator
            if not isinstance(iai_simulator, IAIWrapper):
                iai_simulator = iai_simulator.inner_simulator
            return iai_simulator.inner_simulator.get_state()

        def record_traffic_light_state_history(simulator):
            return simulator.get_traffic_controls()['traffic_light'].state

        record_functions = dict(agent_states=record_agent_states, traffic_light_state_history=record_traffic_light_state_history)
        super().__init__(simulator, record_functions, initial_recording=True)
