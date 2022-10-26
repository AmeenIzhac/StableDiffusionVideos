import Navbar from './Navbar'
import { useState, useEffect } from 'react';
import axios from 'axios';

export default function SimpleUser() {
    const [src, setSrc] = useState("");

    useEffect(() => {
        axios({
            method: 'get',
            url: 'http://109.158.65.154:8080/api',
            responseType: 'blob'
        })
        .then((response) => {
            setSrc(URL.createObjectURL(response.data));
        })
        .catch((error) => {
            console.log(error);
        });
    }, []);
    
    if (src !== "")  {
        return (
            <div className='SimpleUser'>
                <Navbar isSuper={false} link="Super User" href="superUser"/>
                <div className="mainDiv">
                    <div  className='promptDiv'>
                        <input className='prompt' placeholder='Enter Text Prompt...'></input>
                    </div>
                    <div className='videoDiv'>
                        <video id="vidObj" width="500" height="360" controls loop muted autoPlay>
                            <source src={src} type="video/mp4"/>
                        </video>
                    </div>
                </div>
            </div>
        );
    }
  }

  