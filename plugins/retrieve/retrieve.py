# encoding:utf-8

import plugins
from plugins import *
from bridge.context import ContextType

TRIGGER_PREFIX: str = "㵘"

class RetrieveError(Enum):
    COMMAND_NOT_FOUND = 1
    BAD_FORMED_REQUEST = 2

class Command(Enum):
    INSERT = 1
    QUERY = 2
    
def parse_command(s: str) -> Command | RetrieveError:
    match s.casefold():
        case "insert":
            return Command.INSERT 
        case "query":
            return Command.QUERY
        case _:
            return RetrieveError.COMMAND_NOT_FOUND

@plugins.register(name="Retrieve", desire_priority=1, hidden=False, desc="A simple plugin that manages documents", version="0.1", author="jw")
class Retrieve(Plugin):

    def __init__(self):
        super().__init__()
        
        from llama_index.core import Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        Settings.embed_model = HuggingFaceEmbedding()
        Settings.llm = None
        
        from llama_index.core import SimpleDirectoryReader
        self.documents =  SimpleDirectoryReader("./plugins/retrieve/priv").load_data()
        
        from llama_index.core import VectorStoreIndex
        self.vector_index = VectorStoreIndex.from_documents(self.documents)
        
        self.query_engine = self.vector_index.as_query_engine()
        
        self.handlers[Event.ON_HANDLE_CONTEXT] = self.on_handle_context
        logger.info("[retrieve] inited")

    def get_help_text(self, **kwargs):
        return f"""
            This plugin adds RAG ability to this bot.
            
            Commands:
            - {TRIGGER_PREFIX}insert:<insert document>
            - {TRIGGER_PREFIX}query:<query word>
        """
        
    def on_handle_context(self, ctx: EventContext):
        logger.info(f"handling {ctx}")
        if ctx["context"].type != ContextType.TEXT:
            return
        
        content: str = ctx["context"].content
        
        def handle_error(e: RetrieveError):
            ctx.action = EventAction.BREAK_PASS
            ctx["reply"] = f"Retrieve error: {e._name_}"
            logger.warning(f"error happened: {e._name_}")
        
        def handle_result(r: str):
            logger.warning(f"sending to LLM: {r}")
            ctx.action = EventAction.BREAK
            ctx["context"].content = r
        
        match self._on_handle_context_internal(content):
            case RetrieveError(e):
                handle_error(e)
            case str(r):
                handle_result(r)

    def _on_handle_context_internal(self, content: str) -> str | RetrieveError:
        if content.startswith(TRIGGER_PREFIX) == False:
            logger.log(f"other: {content}")
            return
        
        def strip_content(content: str) -> tuple[str, str] | RetrieveError:
            if (deliminator := content.find(":")) != -1:
                return content[1:deliminator], content[(deliminator + 1):]
            return RetrieveError.BAD_FORMED_REQUEST
        
        stripped = strip_content(content)
        
        if stripped is RetrieveError.BAD_FORMED_REQUEST:
            return stripped
        
        command, argument = stripped
        
        match parse_command(command):
            case Command.INSERT:
                return self.do_insertion(argument)
            case Command.QUERY:
                return self.do_query(argument)
            case RetrieveError(r):
                return r

    def do_insertion(self) -> None | RetrieveError:
        raise NotImplementedError()
    
    def do_query(self, query: str) -> str | RetrieveError:
        logger.warning(f"querying keyword {query}")
        return str(self.query_engine.query(query))
