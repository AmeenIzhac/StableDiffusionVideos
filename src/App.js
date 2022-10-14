import  {BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import SimpleUser from './SimpleUser'
import SuperUser from './SuperUser'
import Navbar from './Navbar'

function App() {
  return (
  <>
    <Router>
      <Routes>
        <Route path="" exact element={<SimpleUser />} />
        <Route path="/superUser" exact element={<SuperUser />} />
      </Routes>
    </Router>
  </>
  );
}

export default App;
