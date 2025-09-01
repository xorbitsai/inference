
class ToolParser:
    """
    Abstract ToolParser class that should not be used directly. Provided
    properties and methods should be used in
    derived classes.
    """

    def __init__(self, tokenizer):
        self.prev_tool_call_arr: list[dict] = []
        # the index of the tool call that is currently being parsed
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []

        self.model_tokenizer = tokenizer

    def vocab(self) -> dict[str, int]:
        # NOTE: Only PreTrainedTokenizerFast is guaranteed to have .vocab
        # whereas all tokenizers have .get_vocab()
        return self.model_tokenizer.get_vocab()

    def adjust_request(
            self, request) :
        """
        Static method that used to adjust the request parameters.
        """
        return request

    def extract_tool_calls(
            self, model_output: str,
            request):
        """
        Static method that should be implemented for extracting tool calls from
        a complete model-generated string.
        Used for non-streaming responses where we have the entire model response
        available before sending to the client.
        Static because it's stateless.
        """
        raise NotImplementedError(
            "AbstractToolParser.extract_tool_calls has not been implemented!")

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request ,
    ) :
        """
        Instance method that should be implemented for extracting tool calls
        from an incomplete response; for use when handling tool calls and
        streaming. Has to be an instance method because  it requires state -
        the current tokens/diffs, but also the information about what has
        previously been parsed and extracted (see constructor)
        """
        raise NotImplementedError(
            "AbstractToolParser.extract_tool_calls_streaming has not been "
            "implemented!")