# FireRedTTS3 inference runtime

This directory contains the Base and Instruct inference subset vendored from
[FireRedTeam/FireRedTTS3](https://github.com/FireRedTeam/FireRedTTS3) at commit
`b2ec09a41c3ac89dad8d209391664057a4a1f94b`. The upstream project is licensed
under Apache-2.0.

Only files required by the `FireRedTTS3Base` and `FireRedTTS3Instruct` runtimes
and Xinference's local text frontend are included. Xinference currently exposes
Instruct voice design and voice cloning through the text-to-speech API; the
upstream semantic and acoustic editing methods are vendored for runtime
completeness but are not exposed as Xinference model abilities.
