import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import SimpleUser from "./SimpleUser";
import LoginPage from "./LoginPage";
import SignUpPage from "./SignUpPage";
import Cookies from "js-cookie";
import Gallery from "./Gallery";
// import Navbar from './Navbar'
import { useState } from "react";

function App() {
  const [loggedIn, setLoggedIn] = useState(Cookies.get("loggedInUser") != null);
  return (
    <>
      <Router>
        <Routes>
          <Route
            path="/"
            exact
            element={
              <SimpleUser loggedIn={loggedIn} setLoggedIn={setLoggedIn} />
            }
          />
          <Route
            path="/login"
            element={
              <LoginPage loggedIn={loggedIn} setLoggedIn={setLoggedIn} />
            }
          />
          <Route path="/signup" element={<SignUpPage />} />
          <Route
            path="/gallery"
            element={<Gallery loggedIn={loggedIn} setLoggedIn={setLoggedIn} />}
          />
        </Routes>
      </Router>
    </>
  );
}

export default App;
