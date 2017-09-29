#!/bin/bash

function change_color(){
    find . -name "*.html" -print0 | xargs -0 sed -i -e 's/class="green"/style="color: green; background-color: white;"/g'
    find . -name "*.html" -print0 | xargs -0 sed -i -e 's/class="red"/style="color: red; background-color: white;"/g'
    find . -name "*.html" -print0 | xargs -0 sed -i -e 's/class="gray"/style="color: gray; background-color: white;"/g'
    find . -name "*.html" -print0 | xargs -0 sed -i -e 's/class="orange"/style="color: lightgray; background-color: white;"/g'
}
change_color
