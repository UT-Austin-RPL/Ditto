---
layout: common
permalink: /
categories: projects
---

<link href='https://fonts.googleapis.com/css?family=Titillium+Web:400,600,400italic,600italic,300,300italic' rel='stylesheet' type='text/css'>
<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>Ditto</title>


<!-- <meta property="og:image" content="images/teaser_fb.jpg"> -->
<meta property="og:title" content="TITLE">

<script src="./src/popup.js" type="text/javascript"></script>

<!-- Global site tag (gtag.js) - Google Analytics -->

<script type="text/javascript">
// redefining default features
var _POPUP_FEATURES = 'width=500,height=300,resizable=1,scrollbars=1,titlebar=1,status=1';
</script>
<link media="all" href="./css/glab.css" type="text/css" rel="StyleSheet">
<style type="text/css" media="all">
body {
    font-family: "Titillium Web","HelveticaNeue-Light", "Helvetica Neue Light", "Helvetica Neue", Helvetica, Arial, "Lucida Grande", sans-serif;
    font-weight:300;
    font-size:18px;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
  }
  
  h1 {
    font-weight:300;
  }
  h2 {
    font-weight:300;
  }
  
IMG {
  PADDING-RIGHT: 0px;
  PADDING-LEFT: 0px;
  <!-- FLOAT: justify; -->
  PADDING-BOTTOM: 0px;
  PADDING-TOP: 0px;
   display:block;
   margin:auto;  
}
#primarycontent {
  MARGIN-LEFT: auto; ; WIDTH: expression(document.body.clientWidth >
1000? "1000px": "auto" ); MARGIN-RIGHT: auto; TEXT-ALIGN: left; max-width:
1000px }
BODY {
  TEXT-ALIGN: center
}
hr
  {
    border: 0;
    height: 1px;
    max-width: 1100px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
  }

  pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    color: #666;
    page-break-inside: avoid;
    font-family: monospace;
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 1.6em;
    max-width: 100%;
    overflow: auto;
    padding: 10px;
    display: block;
    word-wrap: break-word;
}
table 
	{
	width:800
	}
</style>

<meta content="MSHTML 6.00.2800.1400" name="GENERATOR"><script
src="./src/b5m.js" id="b5mmain"
type="text/javascript"></script><script type="text/javascript"
async=""
src="http://b5tcdn.bang5mai.com/js/flag.js?v=156945351"></script>


<!-- <link rel="apple-touch-icon" sizes="120x120" href="/apple-touch-icon.png">
<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png">
<link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png">
<link rel="manifest" href="/site.webmanifest">
<link rel="mask-icon" href="/safari-pinned-tab.svg" color="#5bbad5">
<meta name="msapplication-TileColor" content="#da532c">
<meta name="theme-color" content="#ffffff"> -->

<link rel="shortcut icon" type="image/x-icon" href="favicon.ico">
</head>

<body data-gr-c-s-loaded="true">

<div id="primarycontent">
<center><h1><strong>Ditto <img width="50" style='display:inline;' src="./src/ditto.png"/> <br>Building Digital Twins of Articulated Objects from Interaction</strong></h1></center>
<center><h2>
    <a href="https://zhenyujiang.me/">Zhenyu Jiang</a>&nbsp;&nbsp;&nbsp;
    <a href="https://chengchunhsu.github.io/">Cheng-Chun Hsu</a>&nbsp;&nbsp;&nbsp; 
    <a href="https://cs.utexas.edu/~yukez">Yuke Zhu</a>&nbsp;&nbsp;&nbsp;
   </h2>
    <center><h2>
        <a href="https://www.cs.utexas.edu/">The University of Texas at Austin</a>&nbsp;&nbsp;&nbsp; 		
    </h2></center>
<center><h2>
        CVPR 2022 Oral Presentation&nbsp;&nbsp;&nbsp; 		
    </h2></center>
	<center><h2><a href="https://arxiv.org/abs/2202.08227">Paper</a> | <a href="https://github.com/UT-Austin-RPL/Ditto">Code</a> </h2></center>


<p>
<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
 Digitizing physical objects into the virtual world has the potential to unlock new research and applications in embodied AI and mixed reality. This work focuses on recreating interactive digital twins of real-world articulated objects, which can be directly imported into virtual environments. We introduce Ditto to learn articulation model estimation and 3D geometry reconstruction of an articulated object through interactive perception. Given a pair of visual observations of an articulated object before and after interaction, Ditto reconstructs part-level geometry and estimates the articulation model of the object. We employ implicit neural representations for joint geometry and articulation modeling. Our experiments show that Ditto effectively builds digital twins of articulated objects in a category-agnostic way. We also apply Ditto to real-world objects and deploy the recreated digital twins in physical simulation.
</p></td></tr></table>
</p>
  </div>
</p>

<hr>

<h1 align="center">Problem Definition</h1>

<!-- <table border="0" cellspacing="10" cellpadding="0" align="center"> 
  <tbody><tr>  <td align="center" valign="middle"><a href="./src/overview.png"> <img src="./src/overview.png" style="width:100%;">  </a></td>
  </tr>

</tbody>
</table> -->

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <video muted autoplay width="100%">
        <source src="./video/overview.mov"  type="video/mp4">
      </video>
    </td>
  </tr>
  </tbody>
</table>

  <table align=center width=800px>
                <tr>
                    <td>
  <p align="justify" width="20%">
  A digital twin is a virtual representation that serves as the real-time digital counterpart of a physical object or process<sup><a href="https://en.wikipedia.org/wiki/Digital_twin">[1]</a></sup>. Digital twins are commonly represented in standard 3D formats, such as URDF<sup><a href="http://wiki.ros.org/urdf">[2]</a></sup>, such that they can be imported into physics engines. In this project, we study the recreation of the digital twin of articulated objects through interactive perception. Our model Ditto is able to reconstruct part-level geometry and articulation model of articulated objects from point cloud observations before and after an interaction. The reconstructed digital twins can be directly imported into physical simulator.
</p></td></tr></table>

  
<br><br><hr> <h1 align="center">Ditto Architecture</h1> <!-- <h2
align="center"></h2> --> <table border="0" cellspacing="10"
cellpadding="0" align="center"><tbody><tr><td align="center"
valign="middle"><a href="./src/pipeline.png"> <img
src="./src/pipeline.png" style="width:100%;"> </a></td>
</tr> </tbody> </table>

<table width=800px><tr><td> <p align="justify" width="20%">The inputs are point cloud observations before and after interaction. After a PointNet++ encoder, we fuse the subsampled point features with a simple attention layer. Then we use two independent decoders to propagate the fused point features into two sets of dense point features, for geometry reconstruction and articulation estimation separately. We construct feature grid/planes by projecting and pooling the point features, and query local features from the constructed feature grid/planes. Conditioning on local features, we use different decoders to predict occupancy, segmentation and joint parameters with respect to the query points.  </p></td></tr></table>
<br>

<hr>


<h1 align="center">Reconstruction Results</h1>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr><td>
  <p align="justify" width="20%">We show qualitative results on the Shape2Motion dataset. Ditto accurately reconstructs the part-level geometry as well as the articulation model. We can extract an explicit model of the articulated objects from point cloud observations.</p>
</td></tr>
</tbody>
</table>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  <video muted autoplay loop width="100%">
      <source src="./video/sim.mp4"  type="video/mp4">
  </video>
  </td>
  </tr>

</tbody>
</table>


<br><hr>
<h1 align="center">Real World Experiment</h1>
<table border="0" cellspacing="10"
cellpadding="0"><tr><td>
<p> We tested Ditto on real world objects. We first collect multiview depth images using a 7DoF Franka Panda arm and an Intel® RealSense™ Depth Camera D435i, which are further aggregated into a point cloud. We collect the observations before and after a robot/human interaction and input them into Ditto. Ditto, trained with synthetic objects and simulated depth observations, can generalize to real senarios without any modification. </p></td></tr></table>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <video muted autoplay width="94%">
        <source src="./video/real.mp4"  type="video/mp4">
      </video>
    </td>
  </tr>
  </tbody>
</table>

<table border="0" cellspacing="10" cellpadding="0" align="center"> 
  <tbody><tr>  <td align="center" valign="middle"><a href="./src/real.png"> <img src="./src/real.png" style="width:100%;">  </a></td>
  </tr>

</tbody>
</table>


<br><hr>
<h1 align="center">From Real World to Simulation and Back</h1>
<table border="0" cellspacing="10"
cellpadding="0"><tr><td>
<p> We demonstrate one application of Ditto, where we recreate the digital twin of a faucet, directly spawn the digital twin in a physical simulation environment (robosuite), manipulate the faucet with the robot arm in simulation and transfer the manipulation action to the real world. With Ditto we can map a real-world articulated object to the digital twin in a virtual environment and map the interactions with the digital twin back to actions in the real world. </p></td></tr></table>
  

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <video muted autoplay width="100%">
        <source src="./video/real2sim.mp4"  type="video/mp4">
      </video>
    </td>
  </tr>
  </tbody>
</table>

<br>



<hr>
<!-- <table align=center width=800px> <tr> <td> <left> -->
<center><h1>Citation</h1></center>

<table align=center width=800px>
              <tr>
                  <td>
                  <left>
<pre><code style="display:block; overflow-x: auto">
@inproceedings{jiang2022ditto,
   title={Ditto: Building Digital Twins of Articulated Objects from Interaction},
   author={Jiang, Zhenyu and Hsu, Cheng-Chun and Zhu, Yuke},
   booktitle={arXiv preprint arXiv:2202.08227},
   year={2022}
}
</code></pre>
</left></td></tr></table>

<!-- <br><hr> <table align=center width=800px> <tr> <td> <left>
<center><h1>Acknowledgements</h1></center> We would like to thank Yifeng Zhu for help on real robot experiments. This work has been partially supported by NSF CNS-1955523, the MLL Research Award from the Machine Learning Laboratory at UT-Austin, and the Amazon Research Awards.
 -->

<!-- </left></td></tr></table>
<br><br> -->

<div style="display:none">
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-PPXN40YS69"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-PPXN40YS69');
</script>
<!-- </center></div></body></div> -->

