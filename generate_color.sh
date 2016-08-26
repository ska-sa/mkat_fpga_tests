#!/bin/bash

function change_color(){
    echo 'Changing colors on html report'
    find . -name "*.html" -print0 | xargs -0 sed -i -e 's/class="green"/style="color: white; background-color: green;"/g'
    find . -name "*.html" -print0 | xargs -0 sed -i -e 's/class="red"/style="color: white; background-color: red;"/g'
    find . -name "*.html" -print0 | xargs -0 sed -i -e 's/class="gray"/style="color: white; background-color: gray;"/g'
}
change_color
cd ..
change_color
