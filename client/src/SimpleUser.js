import Navbar from "./Navbar";
import { useRef, useState } from "react";
import axios from "axios";
import loadingAnimation from "./assets/loading.gif";
import Cookies from "js-cookie";
import Dropdown from "./Dropdown";
import FrameSelect from "./components/FrameSelect";
import a from "./assets/a.jpg";
import b from "./assets/b.jpg";
import c from "./assets/c.jpg";
import d from "./assets/d.jpg";

export default function SimpleUser() {
  const [src, setSrc] = useState("");
  const promptRef = useRef();
  const [isImg2Img, setisImg2Img] = useState(true);
  const [loading, setLoading] = useState(false);
  const [frames, setFrames] = useState("60");
  const [width, setWidth] = useState("512");
  const [height, setHeight] = useState("512");
  const [angle, setAngle] = useState("0");
  const [fMult, setfMult] = useState("2");
  const [strength, setStrength] = useState("0.4");
  const [zoom, setZoom] = useState("1.1");
  const [fps, setFps] = useState("20");
  const [xShift, setxShift] = useState("0");
  const [yShift, setyShift] = useState("0");
  const [noNoises, setNoNoises] = useState("1");
  const [loggedIn, setLoggedIn] = useState(Cookies.get("loggedInUser") != null);

  //we dont need a state to track the upscaling
  //just use document.getElementById('upscale').checked which will return a boolean

  const [prompts, setPrompts] = useState([]);
  const [progress, setProgress] = useState(0);
  var jobID;
  var fileName = "";

  function resetParams() {
    console.log(`strength is ${strength}`);
    setSrc("");
    setProgress(0);
    setPrompts([]);
    promptRef.current.value = "";
    setisImg2Img(true);
    setFrames("60");
    setWidth("512");
    setHeight("512");
    setAngle("0");
    setZoom("1.1");
    setFps("20");
    setxShift("0");
    setyShift("0");
    setNoNoises("1");
    document.getElementById("upscale").checked = false;
  }

  // Create a new job on server and set the current jobID
  function createJob() {
    if (promptRef.current.value === "") {
      alert("Please enter a prompt");
      return;
    }
    const prompt = promptRef.current.value;
    console.log(`prompt is ${prompt}`);
    fileName = [...prompts, prompt][0].replace(" ", "_");
    setLoading(true);
    axios({
      method: "get",
      url: `https://stablediffusionvideoswebserver-production.up.railway.app/request`,
      // url: `http://localhost:3001/request`,
      params: {
        prompts: (prompt.length === 0
          ? [...prompts]
          : [...prompts, prompt]
        ).join(";"),
        frames: frames,
        width: width,
        height: height,
        angle: angle,
        zoom: zoom,
        fps: fps,
        xShift: xShift,
        yShift: yShift,
        noNoises: noNoises,
        isImg2Img: isImg2Img,
        fMult: fMult,
        strength: strength,
        upscale: document.getElementById("upscale").checked,
      },
      responseType: "application/json",
      timeout: 100000,
    })
      .then((res) => {
        jobID = JSON.parse(res.data).id;
        console.log(jobID);
        poll();
      })
      .catch((err) => {
        console.log(err);
      });

    resetParams();
  }

  async function poll() {
    while (true) {
      const status = await getJobStatus();
      console.log(status);
      switch (status.status) {
        case "pending":
          // setProgress(status.progress)
          break;
        case "generating":
          setProgress(status.progress.progress);
          // setProgress(status.progress)
          break;
        case "done":
          getCreatedVideo();
          return;
        case "error":
          setLoading(false);
          alert("Error generating video");
          return;
        default:
          console.log("Unknown status");
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
  }

  // get status of job
  async function getJobStatus() {
    console.log(jobID);
    return axios({
      method: "get",
      url: `https://stablediffusionvideoswebserver-production.up.railway.app/status`,
      // url: `http://localhost:3001/status`,
      params: {
        jobID: jobID,
      },
      responseType: "text",
      timeout: 10000,
    })
      .then((res) => {
        console.log(JSON.parse(res.data));
        return JSON.parse(res.data);
      })
      .catch((err) => {
        console.log(err);
        return "error";
      });
  }

  function getCreatedVideo() {
    var user = "undefined";
    if (typeof Cookies.get("loggedInUser") != "undefined") {
      user = Cookies.get("loggedInUser");
    }
    axios({
      method: "get",
      url: `https://stablediffusionvideoswebserver-production.up.railway.app/getCreatedVideo`,
      // url: `http://localhost:3001/getCreatedVideo`,
      params: {
        jobID: jobID,
        fileName: fileName,
        user: user,
      },
      responseType: "blob",
      timeout: 10000,
    })
      .then((res) => {
        console.log("res.data");
        console.log(res.data);
        setSrc(URL.createObjectURL(res.data));
        setLoading(false);
      })
      .catch((err) => {
        console.log(err);
        setLoading(false);
      });
  }

  // temp for testing
  const framesT = [a, b, c, d];
  const selectFunction = (index) => {
    console.log(`Frame ${index} Selected`);
  };
  const getNewFrame = () => {
    console.log("Get new frame");
  };
  // end temp testing

  function getVideo() {
    console.log("get viddddddeo");
    const prompt = promptRef.current.value;

    if (prompt === "") {
      alert("Please enter a prompt");
      return;
    }
    setSrc("");
    setLoading(true);
    axios({
      method: "get",
      url: `https://stablediffusionvideoswebserver-production.up.railway.app/generate`,
      // url: `http://localhost:3001/generate`,
      params: {
        prompts: [...prompts, prompt].join(";"),
        frames: frames,
        width: width,
        height: height,
        angle: angle,
        zoom: zoom,
        fps: fps,
        xShift: xShift,
        yShift: yShift,
        noNoises: noNoises,
        isImg2Img: isImg2Img,
        upscale: document.getElementById("upscale").checked,
      },
      responseType: "blob",
      timeout: 10000000,
    })
      .then((response) => {
        console.log(response.data);
        setSrc(URL.createObjectURL(response.data));
        console.log(src);
      })
      .catch((error) => {
        alert("Could not connect to server");
      })
      .finally(() => {
        setLoading(false);
      });

    setAngle("0");
    setZoom("1");
    setWidth("704");
    setHeight("704");

    promptRef.current.value = "";
    setPrompts([]);
    const dropdown = document.getElementById("dropdown");
    const button = document.getElementById("button");
    dropdown.classList.remove("open");
    button.classList.remove("rotate");
  }

  function addPrompt() {
    if (promptRef.current.value === "") {
      alert("Please enter a prompt");
      return;
    }
    setPrompts([...prompts, promptRef.current.value]);
    promptRef.current.value = "";
  }

  //dropdown code
  const dropOptions = () => {
    const dropdown = document.getElementById("dropdown");
    const button = document.getElementById("button");
    dropdown.classList.toggle("open");
    button.classList.toggle("rotate");
  };

  const slideStateChange = (e, set) => {
    set(e.target.value);
  };

  function handleKeyDown(event) {
    if (event.key === "Enter") {
      getVideo();
    }
  }

  return (
    <div className="SimpleUser">
      <Navbar loggedIn={loggedIn} setLoggedIn={setLoggedIn} />
      <Dropdown
        frames={frames}
        setFrames={setFrames}
        isImg2Img={isImg2Img}
        setisImg2Img={setisImg2Img}
        width={width}
        setWidth={setWidth}
        height={height}
        setHeight={setHeight}
        zoom={zoom}
        setZoom={setZoom}
        angle={angle}
        setAngle={setAngle}
        fps={fps}
        setFps={setFps}
        xShift={xShift}
        setxShift={setxShift}
        yShift={yShift}
        setyShift={setyShift}
        noNoises={noNoises}
        setNoNoises={setNoNoises}
        fMult={fMult}
        setfMult={setfMult}
        strength={strength}
        setStrength={setStrength}
        slideStateChange={slideStateChange}
        dropOptions={dropOptions}
      />
      <div className="mainDiv">
        <div className="videoDiv">
          {src ? (
            <video
              id="vidObj"
              width="500"
              height="360"
              controls
              loop
              muted
              autoPlay
            >
              <source src={src} type="video/mp4" />
            </video>
          ) : loading ? (
            <div className="loading">
              <img
                className="loading"
                src={loadingAnimation}
                alt="loading thingy"
              />
              <div className="progress-bar">
                <div
                  className="progress"
                  style={{ width: `${progress * 100}%` }}
                ></div>
              </div>
              <p style={{ color: `var(--main-bg-light)` }}>
                {" "}
                Progress: {Math.round(progress * 10000) / 100}%
              </p>
            </div>
          ) : null}
        </div>
        <div className="promptContainerDiv">
          <div className="promptDiv">
            <input
              className="prompt"
              ref={promptRef}
              placeholder="Enter Text Prompt..."
              onSubmit={createJob}
              onKeyDown={handleKeyDown}
            ></input>
            <button className="promptButton" onClick={addPrompt}>
              +
            </button>
            <button className="promptButton" onClick={createJob}>
              Generate Video
            </button>
            {/* <button className='promptButton' onClick={getVideo} onSubmit={logger}>Generate Video</button> */}
          </div>
          <div className="promptsContainer">
            {prompts.map((prompt, index) => {
              return (
                <div key={index} className="promptsList">
                  <span> {prompt} </span>
                  <button
                    onClick={() => {
                      setPrompts(prompts.filter((_, i) => i !== index));
                    }}
                    className="removePrompt"
                  >
                    <div className="horizontal"></div>
                  </button>
                </div>
              );
            })}
          </div>
        </div>
        {/* <div className='frameDiv'>
          <FrameSelect srcs={frames} selectFunction={selectFunction} getNewFrame={getNewFrame}></FrameSelect>
        </div> */}
      </div>
    </div>
  );
}
