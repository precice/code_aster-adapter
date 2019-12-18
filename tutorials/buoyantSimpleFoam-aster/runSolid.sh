#!/bin/bash

    as_run --quick solid/solid.comm solid/solid.mmed solid/solid.rmed solid/solid.mess --vers=PRECICE > solid.log &
    export PRECICE_PARTICIPANT=Solid
    as_run --run solid/solid.export #> solid.log &
