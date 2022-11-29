import Navbar from "./Navbar";
import { useRef, useState } from "react";
import axios from "axios";
import loadingAnimation from "./assets/loading.gif";
import Cookies from 'js-cookie'
import Dropdown from "./Dropdown";

export default function SimpleUser() {
  const [src, setSrc] = useState("");
  const promptRef = useRef()
  const [loading, setLoading] = useState(false);
  const [frames, setFrames] = useState("30")
  const [width, setWidth] = useState("704")
  const [height, setHeight] = useState("704")
  const [angle, setAngle] = useState("0")
  const [zoom, setZoom] = useState("1")
  const [loggedIn, setLoggedIn] = useState(Cookies.get("loggedInUser") != null)
  const [prompts, setPrompts] = useState([])
  const [progress, setProgress] = useState(0)
  var jobID;

  // Create a new job on server and set the current jobID
  function createJob() {
    const prompt = promptRef.current.value
    setLoading(true);
    axios({
      method: "get",
      // url: `https://stablediffusionvideoswebserver-production.up.railway.app/generate`,
      url: `http://localhost:3001/request`,
      params: {
        prompts: [...prompts, prompt].join(";"),
        frames: frames,
        width: width,
        height: height,
        angle: angle,
        zoom: zoom
      },
      responseType: "application/json",
      timeout: 10000
    })
      .then((res) => {
        jobID = JSON.parse(res.data).id
        console.log(jobID);
        poll();
      })
      .catch((err) => {
        console.log(err);
      });
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
          // setProgress(status.progress)
          break;
        case "done":
          //getVideo();
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
    console.log(jobID)
    return axios({
      method: "get",
      // url: `https://stablediffusionvideoswebserver-production.up.railway.app/status`,
      url: `http://localhost:3001/status`,
      params: {
        jobID: jobID
      },
      responseType: "text",
      timeout: 10000
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

  function getVideo() {
    const prompt = promptRef.current.value

    if (prompt === "") {
      alert("Please enter a prompt");
      return;
    }
    setSrc("");
    setLoading(true);
    axios({
      method: "get",
      // url: `https://stablediffusionvideoswebserver-production.up.railway.app/generate`,
      url: `http://localhost:3001/generate`,
      params: {
        prompts: [...prompts, prompt].join(";"),
        frames: frames,
        width: width,
        height: height,
        angle: angle,
        zoom: zoom
      },
      responseType: "blob",
      timeout: 10000000
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

    setAngle("0")
    setZoom("1")
    setWidth("704")
    setHeight("704")

    promptRef.current.value = "";
    setPrompts([])
    const dropdown = document.getElementById("dropdown")
    const button = document.getElementById("button")
    dropdown.classList.remove("open")
    button.classList.remove("rotate")
  }

  function addPrompt() {
    if (promptRef.current.value === "") {
      alert("Please enter a prompt");
      return;
    }
    setPrompts([...prompts, promptRef.current.value])
    promptRef.current.value = "";
  }

  //dropdown code
  const dropOptions = () => {
    const dropdown = document.getElementById("dropdown")
    const button = document.getElementById("button")
    dropdown.classList.toggle("open")
    button.classList.toggle("rotate")
  }

  const slideFrameChange = (e) => {
    setFrames(e.target.value);
  }

  const slideWidthChange = (e) => {
    setWidth(e.target.value)
  }
  const slideHeightChange = (e) => {
    setHeight(e.target.value)
  }

  const slideAngleChange = (e) => {
    setAngle(e.target.value)
  }

  const slideZoomChange = (e) => {
    setZoom(e.target.value)
  }

  function handleKeyDown(event) {
    if (event.key === 'Enter') {
      getVideo()
    }
  }



  return (
    <div className='SimpleUser'>
      <Navbar loggedIn={loggedIn} setLoggedIn={setLoggedIn} />
      <Dropdown
        frames={frames}
        slideFrameChange={slideFrameChange}
        width={width}
        slideWidthChange={slideWidthChange}
        height={height}
        slideHeightChange={slideHeightChange}
        zoom={zoom}
        slideZoomChange={slideZoomChange}
        angle={angle}
        slideAngleChange={slideAngleChange}
        dropOptions={dropOptions}
      />
      <div className="mainDiv">
        <div className='videoDiv'>
          {src ?
            <video id="vidObj" width="500" height="360" controls loop muted autoPlay>
              <source src={src} type="video/mp4" />
            </video>
            :
            (loading ?
              <img className="loading" src={loadingAnimation} alt='loading thingy' /> : null)
          }
        </div>
        <div className='promptContainerDiv'>
          <div className='promptDiv'>
            <input className='prompt' ref={promptRef} placeholder='Enter Text Prompt...' onSubmit={getVideo} onKeyDown={handleKeyDown}></input>
            <button className='promptButton' onClick={addPrompt}>+</button>
            <button className='promptButton' onClick={getVideo}>Generate Video</button>
            <button className='promptButton' onClick={createJob}>new job</button>
            <button className='promptButton' onClick={getJobStatus}>poll</button>
            {/* <button className='promptButton' onClick={getVideo} onSubmit={logger}>Generate Video</button> */}
          </div>
          <div className="promptsContainer">
            {prompts.map((prompt, index) => {
              return <div key={index} className="promptsList">
                <span> {prompt} </span>
                <button
                  onClick={() => { setPrompts(prompts.filter((_, i) => i !== index)) }}
                  className="removePrompt">
                  <div className="horizontal"></div>
                </button>
              </div>
            })}
          </div>
        </div>
      </div>
    </div>
  );

}
