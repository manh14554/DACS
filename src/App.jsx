import { useState } from "react";
import { TbMovie } from "react-icons/tb";

function App() {
  const [isMovie, setIsMovie] = useState(false)
  const movies = [
    { title: "The Avengers", image: "https://static1.srcdn.com/wordpress/wp-content/uploads/2023/07/blue-beetle-movie-poster.jpg" },
    { title: "Avengers: Age of Ultron", image: "https://static1.srcdn.com/wordpress/wp-content/uploads/2023/07/blue-beetle-movie-poster.jpg" },
    { title: "Iron Man 3", image: "https://static1.srcdn.com/wordpress/wp-content/uploads/2023/07/blue-beetle-movie-poster.jpg" },
    { title: "Captain America: Civil War", image: "https://static1.srcdn.com/wordpress/wp-content/uploads/2023/07/blue-beetle-movie-poster.jpg" },
    { title: "Captain America: The Winter Soldier", image: "https://static1.srcdn.com/wordpress/wp-content/uploads/2023/07/blue-beetle-movie-poster.jpg" },
  ];

  return (
    <div className="d-flex justify-content-center align-items-start min-vh-100 bg-light pt-5">
      <div className="bg-white p-5 rounded shadow w-100" style={{ maxWidth: "1200px" }}>
        <div className="d-flex justify-content-between align-items-center mb-5">
          <h1 className="fs-2 fw-bold mb-0">Hệ thống gợi ý phim</h1>
          <div className="text-secondary">
            <TbMovie style={{ width: "30px", height: "30px" }} />
          </div>
        </div>

        <div className="text-secondary mb-3 fs-4">Hãy lựa chọn bộ phim bạn thích:</div>

        <div className="position-relative mb-5">
          <select className="form-select py-3 fs-5" name="category" id="category">
            <option value="action">Phim hành động</option>
            <option value="blockbuster">Phim bom tấn</option>
          </select>

        </div>

        <button className="btn btn-danger w-100 mb-5 py-3 fs-5"
          onClick={() => { setIsMovie(true) }}>Gợi ý</button>

        <div>
          <div className="fw-semibold mb-3 fs-5">Danh sách phim được gợi ý:</div>
          {isMovie && (<div className="row">
            {movies.map((movie, index) => (
              <div key={index} className="col-6 col-md-3 mb-3">
                <div className="text-center mt-3 fs-5 mb-1">{movie.title}</div>
                <img src={movie.image} alt={movie.title} className="img-fluid rounded" />
              </div>
            ))}
          </div>)}
        </div>
      </div>
    </div>
  );
}

export default App;