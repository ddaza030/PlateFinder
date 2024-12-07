#!/bin/bash
# Inicializa la base de datos y crea la colección "vehiculos"

mongosh <<EOF
use vehiculos_db;
db.createCollection("vehiculos");
EOF