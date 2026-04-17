import { useEffect, useRef, useState } from 'react';
import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isFallDetected, setIsFallDetected] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [facingMode, setFacingMode] = useState('user'); // 'user' is front, 'environment' is back

  const poseLandmarkerRef = useRef(null);
  const requestRef = useRef(null);
  const lastYPositionsRef = useRef([]);

  useEffect(() => {
    let active = true;

    const setupMediaPipe = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm"
        );
        const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numPoses: 1,
        });

        if (active) {
          poseLandmarkerRef.current = poseLandmarker;
          setIsModelLoading(false);
          startWebcam(facingMode);
        }
      } catch (err) {
        console.error("Error loading MediaPipe:", err);
      }
    };

    setupMediaPipe();

    return () => {
      active = false;
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
      if (poseLandmarkerRef.current) poseLandmarkerRef.current.close();
      if (videoRef.current && videoRef.current.srcObject) {
         videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const startWebcam = async (mode) => {
    try {
        // Stop any existing tracks
        if (videoRef.current && videoRef.current.srcObject) {
            videoRef.current.srcObject.getTracks().forEach(track => track.stop());
        }

        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 }, 
                height: { ideal: 480 },
                facingMode: mode
            } 
        });
        if (videoRef.current) {
            videoRef.current.srcObject = stream;
            // Use onloadedmetadata to ensure we have dimensions before starting
            videoRef.current.onloadedmetadata = () => {
                videoRef.current.play();
                predictWebcam();
            };
        }
    } catch (err) {
        console.error("Error accessing webcam: ", err);
    }
  };

  const toggleCamera = () => {
    const newMode = facingMode === 'user' ? 'environment' : 'user';
    setFacingMode(newMode);
    startWebcam(newMode);
  };

  const predictWebcam = async () => {
    if (!videoRef.current || !poseLandmarkerRef.current) return;

    const video = videoRef.current;
    
    // Set canvas dimensions
    if (canvasRef.current && video.videoWidth) {
        canvasRef.current.width = video.videoWidth;
        canvasRef.current.height = video.videoHeight;
    }

    if (video.currentTime > 0) {
        const results = await poseLandmarkerRef.current.detectForVideo(video, performance.now());
        if (results.landmarks && results.landmarks.length > 0) {
            checkFallDetection(results.landmarks[0]);
            drawLandmarks(results.landmarks[0]);
        }
    }

    requestRef.current = requestAnimationFrame(predictWebcam);
  };

  const checkFallDetection = (landmarks) => {
    const leftShoulder = landmarks[11];
    const rightShoulder = landmarks[12];
    const leftHip = landmarks[23];
    const rightHip = landmarks[24];
    
    if (!leftShoulder || !rightShoulder || !leftHip || !rightHip) return;

    // --- 1. Vertical Drop Check ---
    const currentMidY = (leftShoulder.y + rightShoulder.y + leftHip.y + rightHip.y) / 4;
    const history = lastYPositionsRef.current;
    
    history.push({ y: currentMidY, time: performance.now() });
    if (history.length > 30) history.shift(); // keep 30 frames history

    let isSuddenDrop = false;
    if (history.length >= 10) {
        // compare to a point ~10-30 frames ago
        const past = history[0];
        const timeDiffMs = performance.now() - past.time;
        const dropMagnitude = currentMidY - past.y; // Positive = moving down

        // If the center of mass drops substantially inside a small time window
        // Adjust threshold Based on normal speed of standing to floor
        if (dropMagnitude > 0.15 && timeDiffMs < 1500) {
            isSuddenDrop = true;
        }
    }

    // --- 2. Horizontal Posture Check ---
    const shoulderMidX = (leftShoulder.x + rightShoulder.x) / 2;
    const shoulderMidY = (leftShoulder.y + rightShoulder.y) / 2;
    const hipMidX = (leftHip.x + rightHip.x) / 2;
    const hipMidY = (leftHip.y + rightHip.y) / 2;

    const deltaY = Math.abs(shoulderMidY - hipMidY);
    const deltaX = Math.abs(shoulderMidX - hipMidX);
    
    // Ratio of Vertical distance vs Horizontal distance between shoulders & hips
    // When upright, deltaY is much larger than deltaX. 
    // When laying down horizontally, deltaX might be larger or comparable.
    let isHorizontal = false;
    if (deltaY < (deltaX * 1.2)) { // adding 1.2 multiplier to make it slightly more sensitive to horizontal poses
        isHorizontal = true;
    }

    // Also factor in overall vertical distance (if they are squatted, deltaY will be smaller but not < deltaX necessarily)
    // If Sudden Drop or Horizontal Posture is detected -> Set Fall state
    const fallDetectedNow = isSuddenDrop || isHorizontal;

    setIsFallDetected((prev) => {
        if (prev !== fallDetectedNow) return fallDetectedNow;
        return prev;
    });
  };

  const drawLandmarks = (landmarks) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = '#00FF00';
    // Draw landmarks
    for (const landmark of landmarks) {
        ctx.beginPath();
        ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, 4, 0, 2 * Math.PI);
        ctx.fill();
    }
  };

  return (
    <>
      <h1>Fall Detection Assistant</h1>
      {isModelLoading ? (
          <p>Loading Pose Detection Model... Please wait.</p>
      ) : (
          <p>Camera Active. Monitoring for fall events.</p>
      )}

      {isFallDetected && (
          <div className="fall-alert">
              Fall Detected 🚨
          </div>
      )}

      <div className="controls">
        <button onClick={toggleCamera} className="btn-toggle">
            Switch to {facingMode === 'user' ? 'Back' : 'Front'} Camera
        </button>
      </div>

      <div className="video-container">
        <video 
            ref={videoRef} 
            autoPlay 
            playsInline
            muted
            className={`video-feed ${facingMode === 'user' ? 'mirror' : ''}`}
        />
        <canvas 
            ref={canvasRef} 
            className="landmarks-canvas"
        />
      </div>
    </>
  )
}

export default App;
