import threading
import uuid
from collections import OrderedDict
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from config.settings import settings

_MAX_SESSIONS = 1000

class SessionStore:
    def __init__(self, max_turns: int = 6) -> None:
        self._max_messages = max_turns * 2
        self._data : OrderedDict[str, List[BaseMessage]] = OrderedDict()
        self._lock = threading.Lock()

    def new_id(self) -> str:
        return uuid.uuid4().hex

    def get(self, session_id :  str) -> List[BaseMessage]:
        with self._lock:
            return list(self._data.get(session_id,[]))

    def reset(self, session_id : str) -> None:
        with self._lock:
            self._data.pop(session_id, None)

    def append(self, session_id : str, user_msg : str, ai_msg : str) -> None:
        with self._lock:
            history = self._data.get(session_id,[])
            history += [HumanMessage(user_msg), AIMessage(ai_msg)]
            self._data[session_id] = history[-self._max_messages : ]
            self._data.move_to_end(session_id)

            while len(self._data) > _MAX_SESSIONS:
                self._data.popitem(last=False)   

store = SessionStore(max_turns= settings.max_history_turns)