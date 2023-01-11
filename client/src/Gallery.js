import axios from "axios";
import Cookies from "js-cookie";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Gallery.css";

export default function Gallery({ loggedIn, setLoggedIn }) {
  const [videos, setVideos] = useState([]);

  function getGalleryVideos() {
    var user = "undefined";
    if (typeof Cookies.get("loggedInUser") != "undefined") {
      user = Cookies.get("loggedInUser");
    }
    axios({
      method: "get",
      // url: `https://stablediffusionvideoswebserver-production.up.railway.app/getGalleryVideos`,
      // url: `http://localhost:3001/getGalleryVideos`,
      url: "http://18.134.171.110:3001/getGalleryVideos",
      responseType: "json",
      params: {
        user: user,
      },
      timeout: 100000,
    }).then((res) => {
      console.log(res.data.files);
      // setVideo(res.data.files[0]);
      setVideos(res.data.files);
      // setVideo(URL.createObjectURL(vid));
    });
  }
  useEffect(() => {
    getGalleryVideos();
  }, []);

  // function getGalleryVideos() {
  //   var user = "undefined";
  //   if (typeof Cookies.get("loggedInUser") != "undefined") {
  //     user = Cookies.get("loggedInUser");
  //   }
  //   axios({
  //     method: "get",
  //     // url: `https://stablediffusionvideoswebserver-production.up.railway.app/getCreatedVideo`,
  //     url: `http://localhost:3001/getGalleryVideos`,
  //     params: {
  //       user: user,
  //     },
  //     timeout: 100000,
  //   }).then((res) => {
  //     console.log("res");
  //     console.log(res);
  //     console.log("res.data");
  //     console.log(res.data);
  //     console.log("res.data[1]");
  //     console.log(res.data[1]);
  //     console.log("res.data[1][0]");
  //     console.log(res.data[1][0]);
  //     console.log("res.data[1][0].data");
  //     console.log(res.data[1][0].data);

  //     const vid = new Blob(res.data[1][0].data, { type: "video/mp4" });

  //     console.log("vid");
  //     console.log(vid);
  //     console.log("url");
  //     console.log(URL.createObjectURL(vid));
  //     setVideo(URL.createObjectURL(vid));
  //   });
  // }
  const navigate = useNavigate();
  const logout = () => {
    Cookies.remove("loggedInUser");
    setLoggedIn(false);
    navigate("/");
  };

  return (
    <div className="Gallery">
      <div className="Navbar">
        <nav className="navbar">
          <label className="logo">SD VidGen</label>
          <ul>
            <li className="pageLink">
              <a className="link" href="/">
                Home
              </a>
            </li>
            <li className="pageLink">
              <button className="buttonLink" onClick={logout}>
                Logout
              </button>
            </li>
          </ul>
        </nav>
      </div>
      <h1 className="galleryTitle">Gallery</h1>
      <div className="videos">
        {videos.map((videoSrc) => {
          return (
            <div className="vidBox">
              <video className="video" width='40' controls src={videoSrc} />
            </div>
          );
        })}
      </div>
    </div>
  );
}
