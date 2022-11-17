import Navbar from "./Navbar";
import { useState } from "react";
import axios from "axios";
import loadingAnimation from "./assets/painting.gif";
import Cookies from "js-cookie";
import { getStorage, ref, uploadBytes } from "firebase/storage";

export default function SimpleUser() {
  const [src, setSrc] = useState("");
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [frameNumber, setFrameNumber] = useState("30");
  const [width, setWidth] = useState("");
  const [height, setHeight] = useState("");
  const [loggedIn, setLoggedIn] = useState(Cookies.get("loggedInUser") != null);

  const handleChange = (e) => {
    setPrompt(e.target.value);
  };
  function getVideo() {
    if (prompt === "") {
      alert("Please enter a prompt");
      return;
    }

    setLoading(true);
    axios({
      method: "get",
      url: "https://stablediffusionvideoswebserver-production.up.railway.app/api3",
      //   url: "https://stablediffusionvideoswebserver-production.up.railway.app/api",
      //   url: `http://109.158.65.154:8080/api?prompt=` + prompt,
      responseType: "blob",
      timeout: 10000000,
    })
      .then((response) => {
        console.log(response.data);
        setSrc(URL.createObjectURL(response.data));
        const storage = getStorage();
        const user = Cookies.get("loggedInUser");
        const storageRef = ref(storage, { user } + "some-child");

        // 'file' comes from the Blob or File API
        uploadBytes(storageRef, URL.createObjectURL(response.data))
          .then((snapshot) => {
            console.log("Uploaded a blob or file!");
          })
          .then((sth) => {
            console.log("this is the sth" + sth);
          });
        console.log(src);
      })
      .catch((error) => {
        console.log(error);
      })
      .finally(() => {
        setLoading(false);
      });

    setPrompt("");
  }

  //dropdown code
  const dropOptions = () => {
    const dropdown = document.getElementById("dropdown");
    const button = document.getElementById("button");
    dropdown.classList.toggle("open");
    button.classList.toggle("rotate");
  };

  const slideChange = (e) => {
    setFrameNumber(e.target.value);
  };

  const widthChange = (e) => {
    setWidth(e.target.value);
  };
  const heightChange = (e) => {
    setHeight(e.target.value);
  };

  const logger = (e) => {
    e.preventDefault();
    const numberRegex = new RegExp("[0-9]+$");
    console.log(`Prompt is ${prompt}`);
    console.log(`Frames are ${frameNumber}`);
    console.log(`Width is ${width}`);
    if (!(numberRegex.test(width) && numberRegex.test(height))) {
      alert("Please Enter Width and Height as Integer Values");
      return;
    } else {
      const h = parseInt(height);
      const w = parseInt(width);
      if (h % 64 !== 0 || w % 64 !== 0) {
        alert("Width and Height should be multiples of 64");
        return;
      }
    }
    console.log(`Width is ${height}`);
  };

  function handleKeyDown(event) {
    if (event.key === "Enter") {
      logger(event);
    }
  }

  return (
    <div className="SimpleUser">
      <Navbar loggedIn={loggedIn} setLoggedIn={setLoggedIn} />

      <div className="mainDiv">
        <div className="promptContainerDiv">
          <div className="promptDiv">
            <input
              className="prompt"
              value={prompt}
              placeholder="Enter Text Prompt..."
              onChange={handleChange}
              onSubmit={logger}
              onKeyDown={handleKeyDown}
            ></input>
            <button className="promptButton" onClick={getVideo}>
              Generate Video
            </button>
            {/* <button className='promptButton' onClick={getVideo} onSubmit={logger}>Generate Video</button> */}
          </div>
          <div className="slideOptions">
            {Cookies.get("loggedInUser") ? (
              <>
                <div className="dropdownOption" id="dropdown">
                  <form>
                    <div className="slideContainer alignCenter">
                      <p>Number of Frames:</p>
                      <input
                        type="range"
                        min="1"
                        max="60"
                        value={frameNumber}
                        className="slider"
                        id="myRange"
                        onChange={slideChange}
                      />
                      <p>
                        Value: <span id="demo">{frameNumber}</span>
                      </p>
                    </div>
                    <hr />
                    <div className="alignCenter">
                      <input
                        className="dropdownInput alignCenter"
                        value={width}
                        placeholder="Enter Width"
                        onChange={widthChange}
                      />
                    </div>
                    <hr />
                    <div className="alignCenter">
                      <input
                        className="dropdownInput alignCenter"
                        value={height}
                        placeholder="Enter Height"
                        onChange={heightChange}
                      />
                    </div>
                  </form>
                </div>
                <button
                  className="dropArrow"
                  onClick={dropOptions}
                  id="button"
                ></button>
              </>
            ) : (
              <></>
            )}
          </div>
        </div>
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
            <img src={loadingAnimation} alt="loading thingy" />
          ) : null}
        </div>
      </div>
    </div>
  );
}
