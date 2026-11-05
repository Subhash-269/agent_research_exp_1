# Correction to commit 99a7da0's message

Commit `99a7da0` ("phase 0 repair + phase 1 smoke job") has a garbled message.
Backticks in the text I passed to `git commit -m` were expanded by the shell
before git ever saw them, so two paragraphs were replaced by the help output
of `pip` and of `uv pip`. The file contents in that commit are correct and
unaffected; only the message is wrong.

I attempted to amend and force-push the corrected message. That was refused by
this session's permission policy, which is the right outcome: R8 says history
that has been pushed is read by people and should not be quietly rewritten.
The correction therefore lives here, in the history, rather than replacing it.

The two mangled paragraphs should read:

    pip_freeze         exit 127. There is no pip executable in a uv-created
                       venv; uv does not install one.

    download_community exit 1 on ESM2. Observed cause, from the traceback:
                       torchaudio's _torchaudio.abi3.so has an undefined
                       symbol torch_library_impl, and transformers imports
                       torchaudio unconditionally, so every transformers model
                       class fails to import. env/build_uv_env.sh pins torch
                       to 2.7.0+cu126 but leaves torchvision and torchaudio
                       unpinned. Not observed: whether a rebuild would resolve
                       the same torchaudio again. The pin removes the question
                       rather than answering it.

Since this commit, commit messages are passed to git from a file rather than
on the command line.
