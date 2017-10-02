**git-hooks** - A tool to manage project, user, and global Git hooks for multiple git repositories.

git-hooks lets hooks be installed inside git repositories, users home directory, and globally.  
When a hook is called by `git`, git-hooks will check each of these locations for the hooks to run.


Install
=======

Assuming you want to put the hooks in `$pwd/.git/hooks/`:
```
$ chmod +x $pwd/git_hooks/post-commit
$ cd $pwd/git_hooks && bash setup.sh
```


Overview
========

Hooks are powerful and useful.  Some common hooks include:

- Spell check the commit message.
- Verify that the code builds.
- Verify that any new files contain a copyright with the current year in it.

Hooks can be very project-specific such as:

- Verify that the project still builds
- Verify that autotests matching the modified files still pass with no errors.
- Pre-populate the commit message with a "standard" format.
- Verify that any new code follows a "standard" coding style.

Or very person-specific hooks, such as:

- Don't allow a `push` to a remote repository after 1AM, in case I break something and will be asleep.
- Don't allow a commit between 9-5 for projects in `~/personal/`, as I shouldn't be working on them during work hours.

For more details about the different hooks available to you, check out:

       http://www.kernel.org/pub/software/scm/git/docs/githooks.html



Locations
=========

git-hooks provide a way to manage and share your hooks using three locations:

 - **User hooks**, installed in `~/.git/hooks/`
 - **Project hooks**, installed in `git_hooks/` in a project.
 - **Global hooks**,

         $ vim ~/.gitconfig
edit/add templates location
    ```
     [init]
    templatedir = ~/.git-templates
    ```
Global hooks
`~/.git-templates/hooks`

Creating hooks
==============

To keep things organized, git-hooks looks for scripts in **sub-directories** named after the git hook name.  For example, this project has the following `post-commit` script in the following location:

       git_hooks/post-commit/

##see links below
http://githooks.com/
https://git-scm.com/book/gr/v2/Customizing-Git-Git-Hooks