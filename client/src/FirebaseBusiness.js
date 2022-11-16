import firebase from 'firebase/compat/app';
import 'firebase/compat/firestore';

const firebaseConfig = {
    apiKey: "AIzaSyDQu2eHjiw-jjuWM3cU3aoNc8o4Eb4887o",
    authDomain: "stable-diffusion-videos.firebaseapp.com",
    databaseURL: "https://stable-diffusion-videos-default-rtdb.firebaseio.com",
    projectId: "stable-diffusion-videos",
    storageBucket: "stable-diffusion-videos.appspot.com",
    messagingSenderId: "59636912597",
    appId: "1:59636912597:web:a71870a3f93246d1842d5c",
    measurementId: "G-JYZ1L3CB5F"
  };

  firebase.initializeApp(firebaseConfig);

  export default firebase;