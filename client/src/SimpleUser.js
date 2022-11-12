import Navbar from "./Navbar";
import { useState } from "react";
import axios from "axios";
import loadingAnimation from "./assets/painting.gif";

export default function SimpleUser() {
  const [src, setSrc] = useState("");
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);

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
      url: `http://109.158.65.154:8080/api?prompt=` + prompt,
      responseType: "blob",
      timeout: 10000000,
    })
      .then((response) => {
        console.log(response.data);
        setSrc(URL.createObjectURL(response.data));
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

  return (
    <div className="SimpleUser">
      <Navbar isSuper={false} link="Super User" href="superUser" />
      <div className="mainDiv">
        <div className="promptContainerDiv">
          <div className="promptDiv">
            <input
              className="prompt"
              value={prompt}
              placeholder="Enter Text Prompt..."
              onChange={handleChange}
            ></input>
            <button className="promptButton" onClick={getVideo}>
              Generate Video
            </button>
          </div>
          <div className="slideOptions">
            <div className="dropdownOption" id="dropdown">
              hello mate
            </div>
            <button
              className="dropArrow"
              onClick={dropOptions}
              id="button"
            ></button>
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
