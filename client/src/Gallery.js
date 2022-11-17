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
        };
        xhr.open("GET", url);
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
    <div>
      <a href="/">Go back to search</a>
      <h1>Hello World</h1>
      <video src={video} alt="no video for u mate" />
    </div>
  );
}
