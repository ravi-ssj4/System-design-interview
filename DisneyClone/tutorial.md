* layout.tsx: 
    * Top level layout of the app
    * All other views, etc get injected into it

### Next step: break down the UI into components

#### First component: Search
* Note: The structure of components has changed -> now we need to have separate folders for each component instead of just having any files -> each folder has a predefined structure of files -> page.tsx, layout.tsx which we need to fill

* Wildcard for search component -> search/[term]/page.tsx, etc
* Note: 
    * From Nextjs 13 onwards, every component inside the app folder is a ServerComponent.
    * Correct practice: Let them be Server Components and try to convert some of them as granular Client Components whenever needed.

#### Second Component: Genre
* Note: Genre uri structure: http://localhost:3000/genre/80?genre=Crime
* so folder structure inside apps: /apps/genre/[id]/page.tsx, etc

* Note: NextJS 13 onwards, a lot of things are based on the URI
    * Eg. The caching mechanism directly takes the uri param as the key of the cached data

#### Next step: Install shadecn to use their readymade components
* npx shadcn-ui@latest init
* npx shadcn-ui@latest add button
* A new ui folder created inside components folder and all the shadecn components go there -> we can use the code as our own -> build on top of it / customize it

#### Next step: Create Header in the main layout.tsx


```javascript
// /components/Header.tsx
import Image from 'next/image'
import Link from 'next/link'
import React from 'react'

function Header() {
  return (
    <header>
        <Link href="/">
            <Image 
                src="https://links.papareact.com/a943ae"
                alt='Disney Logo'
                height={120}
                width={100}
                className='cursor-pointer invert'
            />
        </Link>
    </header>
  )
}

export default Header

// next.config.mjs
/** @type {import('next').NextConfig} */
const nextConfig = {
    images: {
        remotePatterns: [
            {
                protocol: 'https',
                hostname: 'links.papareact.com'
            }
        ]
    }
};

export default nextConfig;
```

#### Next step: We need to add 3 things to the right of the header:
1. Genre dropdown
2. SearchInput
3. ThemeToggler

#### First we add the theme toggler from shadecn(copy paste):
```javascript
// ThemeToggler.tsx
"use client"

import * as React from "react"
import { Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

export function ThemeToggler() {
  const { setTheme } = useTheme()

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="icon">
          <Sun className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
          <Moon className="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
          <span className="sr-only">Toggle theme</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={() => setTheme("light")}>
          Light
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme("dark")}>
          Dark
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme("system")}>
          System
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

```

#### Add it to our Header
```javascript
import Image from 'next/image'
import Link from 'next/link'
import React from 'react'
import { ThemeToggler } from './ThemeToggler'

function Header() {
    return (
        // here, we also style the header accordingly
        <header className='fixed w-full z-20 top-0 flex items-center 
        justify-between p-5 bg-gradient-to-t from-gray-200/0 
        via-gray-900/25 to-gray-900'>
            
            {/* First part of the header(Left side)
        Disney Logo that takes to the homepage */}

            <Link href="/">
                <Image
                    src="https://links.papareact.com/a943ae"
                    alt='Disney Logo'
                    height={120}
                    width={100}
                    className='cursor-pointer invert-0 dark:invert'
                />
            </Link>

            {/* Second part of the header(right side)
        1. Genre Dropdown
        2. Search input 
        3. Theme toggler */}
            <div className='flex space-x-2'>
                <ThemeToggler />
            </div>
        </header>
    )
}

export default Header

```

#### Next, we work on the SearchInput component(actually a shadecn form)
```js
// SearchInput.tsx
// basically, its a Client Component instead of a server component, 
// as the form is submitted by the client in the browswer
"use client" 
import { zodResolver } from '@hookform/resolvers/zod'
import { useRouter } from 'next/navigation'
import React from 'react'
import { useForm } from 'react-hook-form'
import { z } from "zod"
import { Form, FormControl, FormField, FormItem } from './ui/form'
import { Input } from './ui/input'

const formSchema = z.object({
  input: z.string().min(2).max(50),
})



function SearchInput() {

    const router = useRouter()

    // Copied from shadecn:
    // 1. Define your form.
    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
        input: "",
        },
    })
    
    // 2. Define a submit handler.
    function onSubmit(values: z.infer<typeof formSchema>) {
        // Do something with the form values.
        // ✅ This will be type-safe and validated.
        console.log(values)
        router.push(`/search/${values.input}`)
        form.reset()
    }


  return (
    <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className='space-y-8'>
        <FormField 
            control={form.control}
            name="input"
            render={({ field }) => (
                <FormItem>
                    <FormControl>
                        <Input placeholder="Search..." {...field} />
                    </FormControl>
                </FormItem>
            )}
        />
        </form>
    </Form>
  )
}

export default SearchInput
```

#### Next, we work on the GenreDropdown component -> calls the TMDB's REST API to get movie related data

#### Extra steps:
1. Create typings.ts in the outermost level
```js
export type Movie = {
  adult: boolean;
  backdrop_path: string;
  genre_ids: number[];
  id: number;
  original_language: string;
  original_title: string;
  overview: string;
  popularity: number;
  poster_path?: string;
  release_date: string;
  title: string;
  video: boolean;
  vote_average: number;
  vote_count: number;
};

export type SearchResults = {
  page: number;
  results: Movie[];
  total_pages: number;
  total_results: number;
};

export type Genre = {
  id: number;
  name: string;
};

export type Genres = {
  genres: Genre[];
};

```
2. Create .env.local for env variables, again, outside level
```js
TMDB_API_KEY=secret_key_goes_here

```
3. Create GenreDropdown.tsx
```js
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Genres } from "@/typings";
import { ChevronDown } from "lucide-react";
import Link from "next/link";

async function GenreDropdown() {
  const url = "https://api.themoviedb.org/3/genre/movie/list?language=en";
  const options: RequestInit = {
    method: "GET",
    headers: {
      accept: "application/json",
      Authorization: `Bearer ${process.env.TMDB_API_KEY}`,
    },
    next: {
      revalidate: 60 * 60 * 24, // 24 hours
    },
  };

  const response = await fetch(url.toString(), options);
  const data = (await response.json()) as Genres;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger className="text-white flex justify-center items-center">
        Genre <ChevronDown className="ml-1" />
      </DropdownMenuTrigger>

      <DropdownMenuContent>
        <DropdownMenuLabel>Select a Genre</DropdownMenuLabel>
        <DropdownMenuSeparator />

        {data.genres.map((genre) => (
          <DropdownMenuItem className="cursor-pointer" key={genre.id}>
            <Link href={`/genre/${genre.id}?genre=${genre.name}`}>
              {genre.name}
            </Link>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export default GenreDropdown;

```

#### Next step: add all these components to the Header

#### Next step: Start with the Homepage
```js
import MoviesCarousel from "@/components/MoviesCarousel";

export default async function Home() {
  const upcomingMovies = getUpcomingMovies()
  const topRatedMovies = getTopRatedMovies()
  const popularMovies = getPopularMovies()

  return (
    <main className="">
      <CarouselBannerWrapper />

      <div className="flex flex-col space-y-2 xl:-mt-48">
        <MoviesCarousel title="Upcoming" movies={upcomingMovies} />
        <MoviesCarousel title="Top Rated" movies={topRatedMovies} />
        <MoviesCarousel title="Popular" movies={popularMovies} />
      </div>
    
    </main>    
  );
}

```

#### Backend to fetch data: Create a file inside the lib folder(created when we installed shadecn)
```js
// /lib/getMovies.tsx
import { SearchResults } from "@/typings";

async function fetchFromTMDB(url: URL, cacheTime?: number) {
  url.searchParams.set("include_adult", "false");
  url.searchParams.set("include_video", "false");
  url.searchParams.set("sort_by", "popularity.desc");
  url.searchParams.set("language", "en-US");
  url.searchParams.set("page", "1");

  const options: RequestInit = {
    method: "GET",
    headers: {
      accept: "application/json",
      Authorization: `Bearer ${process.env.TMDB_API_KEY}`,
    },
    next: {
      revalidate: cacheTime || 60 * 60 * 24,
    },
  };

  const response = await fetch(url.toString(), options);
  const data = (await response.json()) as SearchResults;
  return data;
}

export async function getDiscoverMovies(id?: string, keywords?: string) {
  const url = new URL(`https://api.themoviedb.org/3/discover/movie`);

  keywords && url.searchParams.set("with_keywords", keywords);
  id && url.searchParams.set("with_genres", id);

  const data = await fetchFromTMDB(url);
  return data.results;
}

export async function getSearchedMovies(term: string) {
  const url = new URL("https://api.themoviedb.org/3/search/movie");

  url.searchParams.set("query", term);
  url.searchParams.set("include_adult", "false");
  url.searchParams.set("language", "en-US");
  url.searchParams.set("page", "1");

  const options: RequestInit = {
    method: "GET",
    headers: {
      accept: "application/json",
      Authorization: `Bearer ${process.env.TMDB_API_KEY}`,
    },
    next: {
      revalidate: 60 * 60 * 24,
    },
  };

  const response = await fetch(url.toString(), options);
  const data = (await response.json()) as SearchResults;

  return data.results;
}

export async function getUpcomingMovies() {
  const url = new URL("https://api.themoviedb.org/3/movie/upcoming");
  const data = await fetchFromTMDB(url);

  return data.results;
}

export async function getTopRatedMovies() {
  const url = new URL("https://api.themoviedb.org/3/movie/top_rated");
  const data = await fetchFromTMDB(url);

  return data.results;
}

export async function getPopularMovies() {
  const url = new URL("https://api.themoviedb.org/3/movie/popular");
  const data = await fetchFromTMDB(url);

  return data.results;
}

```