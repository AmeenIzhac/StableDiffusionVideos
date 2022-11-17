import "./Gallery.css";

import { useState, useEffect } from "react";
import { getStorage, ref, getDownloadURL } from "firebase/storage";

export default function Gallery() {
  const [video, setVideo] = useState("");

  useEffect(() => {
    const storage = getStorage();
    getDownloadURL(ref(storage, "some-child"))
      .then((url) => {
        const xhr = new XMLHttpRequest();
        xhr.responseType = "blob";
        xhr.onload = (event) => {
          setVideo(URL.createObjectURL(xhr.response));
          console.log(video);
        };
        xhr.open("GET", url);
        xhr.setRequestHeader("Access-Control-Allow-Origin", "*");
        xhr.send();
        // Or inserted into an <img> element
        const img = document.getElementById("myimg");
        img.setAttribute("src", url);
      })
      .catch((error) => {
        // Handle any errors
      });
  }, []);
  return (
    <div className="Gallery">
      <a href="/">Go back to search</a>
      <br />
      <h1 className="title">Your Gallery</h1>
      <div className="vidsContainer">
        <video
          src={video ? video : "vid2.mp4"}
          className="vid"
          alt="no video for u mate"
        />
        <video src="vid2.mp4" className="vid" alt="no video for u mate" />
        <video src="vid3.mp4" className="vid" alt="no video for u mate" />
        <video src="vid4.mp4" className="vid" alt="no video for u mate" />
        <video src="vid1.mp4" className="vid" alt="no video for u mate" />
        <video src="vid2.mp4" className="vid" alt="no video for u mate" />
        <video src="vid3.mp4" className="vid" alt="no video for u mate" />
        <video src="vid4.mp4" className="vid" alt="no video for u mate" />
        <video src="vid1.mp4" className="vid" alt="no video for u mate" />
        <video src="vid2.mp4" className="vid" alt="no video for u mate" />
        <video src="vid3.mp4" className="vid" alt="no video for u mate" />
        <video src="vid4.mp4" className="vid" alt="no video for u mate" />
        <video src="vid1.mp4" className="vid" alt="no video for u mate" />
        <video src="vid2.mp4" className="vid" alt="no video for u mate" />
        <video src="vid3.mp4" className="vid" alt="no video for u mate" />
        <video src="vid4.mp4" className="vid" alt="no video for u mate" />
        <video src="vid1.mp4" className="vid" alt="no video for u mate" />
        <video src="vid2.mp4" className="vid" alt="no video for u mate" />
        <video src="vid3.mp4" className="vid" alt="no video for u mate" />
        <video src="vid4.mp4" className="vid" alt="no video for u mate" />
        <video src="vid1.mp4" className="vid" alt="no video for u mate" />
        <video src="vid2.mp4" className="vid" alt="no video for u mate" />
        <video src="vid3.mp4" className="vid" alt="no video for u mate" />
        <video src="vid4.mp4" className="vid" alt="no video for u mate" />
        <video src="vid1.mp4" className="vid" alt="no video for u mate" />
        <video src="vid2.mp4" className="vid" alt="no video for u mate" />
        <video src="vid3.mp4" className="vid" alt="no video for u mate" />
        <video src="vid4.mp4" className="vid" alt="no video for u mate" />
        <video src="vid1.mp4" className="vid" alt="no video for u mate" />
        <video src="vid2.mp4" className="vid" alt="no video for u mate" />
        <video src="vid3.mp4" className="vid" alt="no video for u mate" />
        <video src="vid4.mp4" className="vid" alt="no video for u mate" />
        <video src="vid1.mp4" className="vid" alt="no video for u mate" />
        <video src="vid2.mp4" className="vid" alt="no video for u mate" />
        <video src="vid3.mp4" className="vid" alt="no video for u mate" />
        <video src="vid4.mp4" className="vid" alt="no video for u mate" />
      </div>
    </div>
  );
}
