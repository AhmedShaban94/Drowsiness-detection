#include <Windows.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#pragma comment(lib, "winmm.lib")

#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\opencv.hpp"

#include "dlib\gui_widgets.h"
#include "dlib\image_io.h"
#include "dlib\image_processing.h"
#include "dlib\image_processing\frontal_face_detector.h"
#include "dlib\image_processing\render_face_detections.h"
#include "dlib\opencv.h"

void sound_alarm(const std::string& alarmDir)
{
    PlaySound(TEXT(alarmDir.c_str()), NULL, SND_FILENAME);
}

double eye_aspect_ratio(const std::vector<dlib::point>& eye)
{
    // aspect ratio will be near zero during the blink.

    // calculate Ecludian distance between two sets of vertical eye landmarks.
    const auto A = (eye[1] - eye[5]).length_squared();
    const auto B = (eye[2] - eye[4]).length_squared();

    // calculate Ecludian distance between horizontal eye landmarks.
    const auto C = (eye[0] - eye[3]).length_squared();

    return ((A + B) / (2.0 * C));
}

int main(void)
{
    // eye aspect ratio thershold to detect eye blink.
    constexpr double EYE_ASPECT_RATIO_THRESHOLD = 0.09;
    // # of consecutive frames the eye should be below the aspect ratio
    // threshold to sound alarm
    constexpr int EYE_ASPECT_RATIO_CONSECUTIVE_FRAMES = 48;

    // counter indicates the number of frames the eye closed.
    int counter = 0;

    // bool indicates if the eye is closed or not.
    bool alarm_on = false;
    // directory to sound alarm file.

    const std::string alarmdDir = R"(utils\alarm.wav)";
    // directory to shape predictor pretrained model.
    const std::string shapePredictorDir
        = R"(utils\shape_predictor_68_face_landmarks.dat)";

    // init stream from webCam.
    auto stream = std::make_unique<cv::VideoCapture>(0);
    if (!stream->isOpened())
    {
        std::cerr << "Can't connect to camera\n";
        return EXIT_FAILURE;
    }

    // init face detector & shape predictor
    auto detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor predictor{};
    dlib::deserialize(shapePredictorDir) >> predictor;

    while (cv::waitKey(30) != 27) // 27 is the ascii code for ESC key.
    {
        // fetch each frame & convert it to dlib_frame.
        cv::Mat frame, grey;
        stream->read(frame);
        cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
        dlib::cv_image<uchar> dlib_frame(grey);

        // detect faces in each frame.
        std::vector<dlib::rectangle> faces = detector(dlib_frame);

        // find the pose of each face
        std::vector<dlib::full_object_detection> shapes;
        for (const auto& face : faces)
        {
            shapes.push_back(predictor(dlib_frame, face));

            std::vector<dlib::point> leftEye{
                shapes[0].part(36), shapes[0].part(37), shapes[0].part(38),
                shapes[0].part(39), shapes[0].part(40), shapes[0].part(41)
            };

            std::vector<dlib::point> rightEye{
                shapes[0].part(42), shapes[0].part(43), shapes[0].part(44),
                shapes[0].part(45), shapes[0].part(46), shapes[0].part(47)
            };

            const auto leftAspectRatio  = eye_aspect_ratio(leftEye);
            const auto rightAspectRatio = eye_aspect_ratio(rightEye);
            const auto averageAspectRatio
                = static_cast<double>((leftAspectRatio + rightAspectRatio) / 2);
            std::cout << "Eye aspect ratio: " << averageAspectRatio << '\n';

            // convert dlib points to cv point in order to calculate convex hull
            std::vector<cv::Point> leftEyeCV, rightEyeCV;
            for (const auto& dlib_point : leftEye)
                leftEyeCV.push_back(cv::Point(dlib_point.x(), dlib_point.y()));
            for (const auto& dlib_point : rightEye)
                rightEyeCV.push_back(cv::Point(dlib_point.x(), dlib_point.y()));

            std::vector<cv::Point> leftHull, rightHull;
            cv::convexHull(cv::Mat(leftEyeCV), leftHull, false);
            cv::convexHull(cv::Mat(rightEyeCV), rightHull, false);

            // draw eyes contours to frame.
            cv::Scalar color(0, 0, 255);
            drawContours(frame, std::vector<std::vector<cv::Point>>{ leftHull },
                         -1, color, 1);
            drawContours(frame,
                         std::vector<std::vector<cv::Point>>{ rightHull }, -1,
                         color, 1);

            if (averageAspectRatio < EYE_ASPECT_RATIO_THRESHOLD)
            {
                counter++;
                if (counter >= EYE_ASPECT_RATIO_CONSECUTIVE_FRAMES)
                {
                    if (!alarm_on)
                        alarm_on = true;
                    if (alarm_on)
                    {
                        cv::putText(frame, "DROWSINESS ALERT!",
                                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                                    0.7, cv::Scalar(0, 0, 255), 2);
                        sound_alarm(alarmdDir);
                    }
                }
            }
            else
            {
                alarm_on = false;
                counter  = 0;
            }
        }
        cv::imshow("video", frame);
    }
    stream.release();
    cv::destroyAllWindows();
    // sound_alarm(alarmdDir);
    return EXIT_SUCCESS;
}