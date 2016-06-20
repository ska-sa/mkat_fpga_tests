"""Drop into an embedded IPython session for debugging purposes."""
from IPython.terminal.embed import InteractiveShellEmbed
# from IPython.config.loader import Config
import inspect
import traitlets

# Configure the prompt so that I know I am in a nested (embedded) shell
cfg = traitlets.config.Config()
# cfg = Config()
prompt_config = cfg.PromptManager
prompt_config.in_template = 'Debug.In [\\#]: '
prompt_config.in2_template = '   .\\D.: '
prompt_config.out_template = 'Debug.Out[\\#]: '

# Messages displayed when I drop into and exit the shell.
banner_msg = ("\n**Embedded Interpreter:\n"
              "Hit Ctrl-D to exit interpreter and continue program.\n"
              "Note that if you use %kill_embedded, you can fully deactivate\n"
              "This embedded instance so it will never turn on again")
exit_msg = '**Leaving Embedded interpreter'


# Wrap it in a function that gives me more context:
def IPythonShell():
    ipshell = InteractiveShellEmbed(config=cfg, banner1=banner_msg, exit_msg=exit_msg)

    frame = inspect.currentframe().f_back
    msg = 'Stopped at {0.f_code.co_filename} at line {0.f_lineno}'.format(frame)

    # Go back one level!
    # This is needed because the call to ipshell is inside the function ipsh()
    ipshell(msg, stack_depth=2)
