#!/bin/bash

    cd fluid; blockMesh; cd ..
    buoyantSimpleFoam -case fluid
