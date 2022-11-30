import Navbar from "./Navbar";
import { useRef, useState } from "react";
import axios from "axios";
import loadingAnimation from "./assets/loading.gif";
import Cookies from 'js-cookie'
import Dropdown from "./Dropdown";

export default function SimpleUser() {
  const [src, setSrc] = useState("");
  const promptRef = useRef()
  const [isImg2Img, setisImg2Img] = useState(true);
  const [isWalk, setisWalk] = useState(false)
  const [loading, setLoading] = useState(false);
  const [frames, setFrames] = useState("60")
  const [width, setWidth] = useState("512")
  const [height, setHeight] = useState("512")
  const [angle, setAngle] = useState("0")
  const [zoom, setZoom] = useState("1.1")
  const [fps, setFps] = useState("20")
  const [xShift,setxShift] = useState("1")
  const [yShift,setyShift] = useState("1")
  const [noNoises, setNoNoises] = useState("1")
  const [loggedIn, setLoggedIn] = useState(Cookies.get("loggedInUser") != null)

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
      url: `https://stablediffusionvideoswebserver-production.up.railway.app/generate?prompt=${prompt}&frames=${frames}&width=${width}&height=${height}&angle=${angle}&zoom=${zoom}`,
      // url: `http://localhost:3001/generate?prompt=${prompt}&frames=${frames}&width=${width}&height=${height}`,
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

    setAngle("0")
    setZoom("1")
    setWidth("704")
    setHeight("704")
    const dropdown = document.getElementById("dropdown")
    const button = document.getElementById("button")
    dropdown.classList.remove("open")
    button.classList.remove("rotate")
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

  const slideFpsChange = (e) => {
    setFps(e.target.value)
  }
  const slidexShiftChange = (e) => {
    setxShift(e.target.value)
  }
  const slideyShiftChange = (e) => {
    setyShift(e.target.value)
  }
  const slideNoNoisesChange = (e) => {
    setNoNoises(e.target.value)
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
        isImg2Img={isImg2Img}
        setisImg2Img={setisImg2Img}
        isWalk={isWalk}
        setisWalk={setisWalk}
        width={width}
        slideWidthChange={slideWidthChange}
        height={height}
        slideHeightChange={slideHeightChange}
        zoom={zoom}
        slideZoomChange={slideZoomChange}
        angle={angle}
        slideAngleChange={slideAngleChange}
        fps={fps}
        slideFpsChange={slideFpsChange}
        xShift={xShift}
        slidexShiftChange={slidexShiftChange}
        yShift={yShift}
        slideyShiftChange={slideyShiftChange}
        noNoises={noNoises}
        slideNoNoisesChange={slideNoNoisesChange}
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
            <button className='promptButton' onClick={getVideo}>Generate Video</button>
            {/* <button className='promptButton' onClick={getVideo} onSubmit={logger}>Generate Video</button> */}
          </div>
            
        </div>
      </div>
    </div>
  );
}

