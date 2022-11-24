import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import SimpleUser from './SimpleUser'
import LoginPage from './LoginPage'
import SignUpPage from './SignUpPage'
import Gallery from './Gallery'
// import Navbar from './Navbar'

function App() {
  return (
  <>
    <Router>
      <Routes>
        <Route path="/" exact element={<SimpleUser />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/signup" element={<SignUpPage />} />
        <Route path="/gallery" element={<Gallery />} />
      </Routes>
    </Router>
  </>
  );
}

export default App;
