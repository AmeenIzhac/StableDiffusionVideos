import Navbar from './Navbar'
import { useState, useEffect } from 'react';
import axios from 'axios';

export default function SimpleUser() {

    const [test, setTest] = useState("Hello World");

    useEffect(() => { 
        console.log("axios call");
        axios.get('http://109.158.65.154:8080/api').then((response) => {
            console.log(response.data[0].prompt); 
        });
        console.log("axios call end");
    }, []);

    return (
        <div className='SimpleUser'>
            <Navbar isSuper={false} link="Super User" href="superUser"/>
            <div className="mainDiv">
                <div  className='promptDiv'>
                    <input className='prompt' placeholder='Enter Text Prompt...'></input>
                </div>
                <div className='videoDiv'>
                    <video className="video" controls>
                        <source src="vid.mp4" type="video/mp4"/>
                        Your browser does not support the video tag.
                    </video>
                </div>
                <div>
                    <p>{test}</p>
                </div>
            </div>
        </div>
    );
  }

  