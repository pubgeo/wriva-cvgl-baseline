<h3>Metric Evaluation</h3>
<p>

Given multiple groups of sequential ground-level images and one or more satellite images covering the full site, locate each camera accurately. Ground-level images are intentionally placed such that camera calibration of all images in a scene without cross-view matching using the satellite images is difficult. Multiple datasets of varying difficulty will be provided.

* Inputs for each dataset
  * Unposed ground-level images
  * Multiple orthorectified Maxar satellite images with GeoTIFF metadata
* Output for each dataset
  * JSON text file including camera locations for each input ground-level image
* Evaluation metric for ground-level camera geo-localization
  * Horizontal position error (meters), ninetieth percentile for each dataset, averaged over all datasets and reported with two significant digits
* The leaderboard metric may be refined throughout the development phase in advance of launching the test phase

</p>