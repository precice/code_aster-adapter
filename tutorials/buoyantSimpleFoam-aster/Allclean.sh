#!/bin/bash
rm -f precice-Fluid-convergence.log
rm -f precice-Fluid-events.json
rm -f precice-Fluid-iterations.log
rm -f solid/solid.mess
rm -f solid/solid.resu

# Add file deletion for solid.rmed

# pyFoamClearCase.py

rm -r fluid/constant/polyMesh
rm -r fluid/10
rm -r fluid/20
rm -r fluid/30
