#!/bin/bash
RUST_BACKTRACE=1 cargo watch -x 'test -- --nocapture' -s 'xdotool search --desktop 1 --name "Private Browsing" key --window %@ "CTRL+F5"'
