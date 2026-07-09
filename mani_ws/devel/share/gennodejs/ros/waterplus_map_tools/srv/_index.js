
"use strict";

let GetNumOfWaypoints = require('./GetNumOfWaypoints.js')
let SaveWaypoints = require('./SaveWaypoints.js')
let GetChargerByName = require('./GetChargerByName.js')
let GetWaypointByIndex = require('./GetWaypointByIndex.js')
let GetWaypointByName = require('./GetWaypointByName.js')
let AddNewWaypoint = require('./AddNewWaypoint.js')

module.exports = {
  GetNumOfWaypoints: GetNumOfWaypoints,
  SaveWaypoints: SaveWaypoints,
  GetChargerByName: GetChargerByName,
  GetWaypointByIndex: GetWaypointByIndex,
  GetWaypointByName: GetWaypointByName,
  AddNewWaypoint: AddNewWaypoint,
};
