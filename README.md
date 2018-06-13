# pedestrian-detector
Detects pedestrians in webcam live stream or video.

Results are transformed in xy-plane using a homography.

Activity in defined area is monitored

Contains 3 algorithms (they all do more or less the same):
## Yolo v2
Best results but also the slowest

(heavily influence by [this](https://github.com/devicehive/devicehive-video-analysis)
Requiers:
* download and extract [data.tar.gz](https://s3.amazonaws.com/video-analysis-demo/data.tar.gz) to source folder

See [YouTube](https://www.youtube.com/watch?v=T1NMpha9mFI) for Results.

Also [TownCenterXVID.avi](https://youtu.be/11oC8gntF8Q)
## HoG and SVM
Includes non-maximum suppression

See [YouTube](https://www.youtube.com/watch?v=T1NMpha9mFI) for Results.

Also [TownCenterXVID.avi](https://youtu.be/oVlLKOksvvA)

## Haar features and Cascade 
Fast but inaccurate.

See [YouTube](https://youtu.be/oCIKL3vnTAg)

Also [TownCenterXVID.avi](https://youtu.be/SGiBV-dLM0g)

no backwards homography yet

Includes non-maximum suppression.





