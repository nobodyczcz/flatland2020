//console.log("currentScript:", document.currentScript);

var script_tag = document.getElementById('graph2d');

var graph_file = script_tag.getAttribute("graph_file");
var width = script_tag.getAttribute("width");
var height = script_tag.getAttribute("height");

console.log("inputs:", graph_file, width, height);

//graph_file="./test1.json";

// adapted from: https://github.com/ipython-books/cookbook-2nd-code/blob/master/chapter06_viz/04_d3.ipynb
// We load the d3.js library from the Web.
require.config({paths:
    {d3: "http://d3js.org/d3.v3.min"}});
require(["d3"], function(d3) {
  // The code in this block is executed when the
  // d3.js library has been loaded.

  console.log(document.currentScript);

  // specify the size of the canvas
  // containing the visualization (size of the
  // <div> element).

  if (!width) width = 400;
  if (!height) height = 300;

  // We create a color scale.
  var color = d3.scale.category10();

  // We create a force-directed dynamic graph layout.
  var force = d3.layout.force()
    .charge(function(node) { 
        if (node.type=="rail") return -100;
        else return -100;
        })  // -120
    .linkDistance(50)
    .size([width, height]);

  // In the <div> element, we create a <svg> graphic
  // that will contain our interactive visualization.
  var svg = d3.select("#d3-example").select("svg")
  if (svg.empty()) {
    svg = d3.select("#d3-example").append("svg")
          .attr("width", width)
          .attr("height", height);
  }

  svg.append("svg:defs").selectAll("marker")
  .data(["end"])      // Different link/path types can be defined here
.enter().append("svg:marker")    // This section adds in the arrows
  .attr("id", String)
  .attr("viewBox", "0 -5 10 10")
  .attr("refX", 15)
  .attr("refY", -1.5)
  .attr("markerWidth", 6)
  .attr("markerHeight", 6)
  .attr("orient", "auto")
.append("svg:path")
  .attr("d", "M0,-5L10,0L0,5");

  // We load the JSON file.
  d3.json(graph_file, function(error, graph) {
    // In this block, the file has been loaded
    // and the 'graph' object contains our graph.

    // We load the nodes and links in the
    // force-directed graph.
    var oForce = force.nodes(graph.nodes)
      .links(graph.links);
    
    // Set the linkDistance to a function of a link
    force.linkDistance(function(link) {
       return link.type == "hold" ? 2 : 30;
    });
      
    oForce.start();

    // We create a <line> SVG element for each link
    // in the graph.
    var link = svg.selectAll(".link")
      .data(graph.links)
      .enter().append("line")  // either line or path (for arc curved links)
      // .attr("fill", "#f00")
      .attr("class", "link")
      .style("stroke", function(d) {
          if (d.type == null) return ("#ffffff00")   // invisible if null type
          else return("#ccc")
      })
      // .style("fill", "#ffffff00");  // don't fill arcs...
      .attr("marker-end", function(d) {  // only get an arrow head if type non-null 
        if (d.type == null) return (null);
        else if (d.type == "hold") return (null);
        else return("url(#end)")
        });
      
      // "url(#end)");
    
    //function(d) { 
    //      if (d.type==0) return "#ffffff00";
    //      else return color(d.type);
     //     });

    // We create a <circle> SVG element for each node
    // in the graph, and we specify a few attributes.
    var node = svg.selectAll(".node")
      .data(graph.nodes)
      .enter().append("circle")
      .attr("class", "node")
      .attr("r", 5)  // radius
      .style("fill", // "#f00000")
             function(d) {
                 if (d.type == null) return "#ccc";
                 // The node color depends on the type.
                return color(d.type);
              })
      .call(force.drag);

    // The name of each node is the node title (stored as a tuple).
    node.append("title")
        .text(function(d) { 
            // return d.name; 
            return d.title; 
            });

    // We bind the positions of the SVG elements
    // to the positions of the dynamic force-directed
    // graph, at each time step.
    force.on("tick",  tick0);
    
    // straight line links
    function tick0() {
      link.attr("x1", function(d){return d.source.x})
          .attr("y1", function(d){return d.source.y})
          .attr("x2", function(d){return d.target.x})
          .attr("y2", function(d){return d.target.y})
          .attr("distance", function(d){return d.distance });

      node.attr("cx", function(d){return d.x})
          .attr("cy", function(d){return d.y});
        
    };
    
    // curved links using svg Arc - not currently used!
    function tick1() {
        link.attr("d", function(d) {
        var dx = d.target.x - d.source.x,
            dy = d.target.y - d.source.y,
            dr = Math.sqrt(dx * dx + dy * dy);
    
        var arc1 = 0;
        if (d.type==1) { arc1=1; dr = 1000; }
        
        // M = move to source
        // A = draw arc to target - https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Paths
        // arc1 is the sweep flag 
        return "M" + 
            d.source.x + "," + 
            d.source.y + 
            "A" + dr + "," + dr + " 0 0 " + arc1 + " "+
            d.target.x + "," + 
            d.target.y;
        });
        
        node.attr("cx", function(d){return d.x})
            .attr("cy", function(d){return d.y});

    
    };

  });
});